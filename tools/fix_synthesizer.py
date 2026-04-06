"""
Fix Synthesizer — generates chosen (correct) and rejected (wrong) pandas code
from a mutation map produced by the generator.

WHY THIS EXISTS
---------------
DPO training needs pairs: (prompt, chosen_code, rejected_code).
The model learns: "chosen_code is better than rejected_code for this prompt."

chosen_code  = correct pandas that fixes every broken cell
rejected_code = plausible-looking code with deliberate mistakes

We derive both deterministically from error_map.json — no LLM needed,
no guessing. The mutation map tells us exactly what broke and how severe it is.

HOW IT WORKS
------------
Step 1: Parse the error_map → group errors by (column, corruption_type)
Step 2: For each group, look up the correct fix template → chosen_code
Step 3: For each group, apply a realistic mistake → rejected_code
Step 4: Return both as runnable Python strings

Corruption types handled:
    null_injected       → fillna(median) or fillna(mode)
    type_mangled        → pd.to_numeric + fillna
    duplicate_rows      → drop_duplicates + reset_index
    whitespace_noise    → str.strip()
    format_inconsistency→ str.upper() / lower() / title()
    outlier_injected    → clip(lower, upper) from clean data bounds

Common mistakes injected into rejected_code:
    null_injected       → fillna(0) — zeros are wrong for Age/pH
    type_mangled        → to_numeric but forget fillna — leaves NaNs
    duplicate_rows      → drop_duplicates but forget reset_index
    whitespace_noise    → str.lstrip() only — misses trailing spaces
    format_inconsistency→ wrong case (always lower)
    outlier_injected    → only clip lower bound — misses upper outliers
"""

from __future__ import annotations

from typing import Any

import pandas as pd


# ── Helpers ──────────────────────────────────────────────────────────────────


def _is_numeric(clean_df: pd.DataFrame, col: str) -> bool:
    """Check if a column in the clean data is numeric."""
    if col not in clean_df.columns:
        return False
    return pd.api.types.is_numeric_dtype(clean_df[col])


def _detect_case(clean_df: pd.DataFrame, col: str) -> str:
    """
    Infer the intended casing of a string column from the clean data.
    Returns 'upper', 'lower', or 'title'.
    This matters for format_inconsistency fixes — we restore the original case.
    """
    if col not in clean_df.columns:
        return "title"
    vals = clean_df[col].dropna().astype(str)
    if vals.empty:
        return "title"
    if vals.str.isupper().all():
        return "upper"
    if vals.str.islower().all():
        return "lower"
    return "title"


def _group_errors(error_map: dict[str, Any]) -> dict[str, list[str]]:
    """
    Parse error_map → {corruption_type: [col, col, ...]}

    This tells us: "which columns need which kind of fix."
    For example: {"null_injected": ["Age", "Embarked"], "type_mangled": ["Fare"]}

    Duplicate rows are stored separately in error_map["spurious_rows"],
    not in cell_errors, so we check that separately.
    """
    groups: dict[str, list[str]] = {}

    for key, info in error_map.get("cell_errors", {}).items():
        col = key.rsplit(",", 1)[1]           # "27,Age" → "Age"
        corruption = info["corruption"]       # "null_injected"
        if corruption not in groups:
            groups[corruption] = []
        if col not in groups[corruption]:
            groups[corruption].append(col)

    # Duplicate rows live in spurious_rows, not cell_errors
    if error_map.get("spurious_rows"):
        groups["duplicate_rows"] = []        # no column — row-level fix

    return groups


def _clean_col_bounds(clean_df: pd.DataFrame, col: str) -> tuple[float, float]:
    """Get the min/max of a column from the clean data — used for outlier clipping."""
    numeric = pd.to_numeric(clean_df[col], errors="coerce").dropna()
    return float(numeric.min()), float(numeric.max())


# ── Chosen fix templates (correct) ───────────────────────────────────────────


def _chosen_nulls(cols: list[str], clean_df: pd.DataFrame) -> list[str]:
    """
    For null-injected columns:
    - Numeric cols → fillna(median)  — robust to skew, always in valid range
    - Categorical cols → fillna(mode) — most common value, always valid
    """
    lines = []
    for col in cols:
        if _is_numeric(clean_df, col):
            lines.append(f"df['{col}'] = df['{col}'].fillna(df['{col}'].median())")
        else:
            lines.append(f"df['{col}'] = df['{col}'].fillna(df['{col}'].mode()[0])")
    return lines


def _chosen_type_mangle(cols: list[str], clean_df: pd.DataFrame) -> list[str]:
    """
    For type-mangled columns (garbage strings injected into numeric columns):
    1. pd.to_numeric(..., errors='coerce') — converts garbage → NaN
    2. fillna(median) — fill the NaNs that coerce just created
    """
    lines = []
    for col in cols:
        lines.append(f"df['{col}'] = pd.to_numeric(df['{col}'], errors='coerce')")
        lines.append(f"df['{col}'] = df['{col}'].fillna(df['{col}'].median())")
    return lines


def _chosen_duplicates() -> list[str]:
    """
    For duplicate rows: drop them and reset the integer index.
    reset_index(drop=True) is critical — without it the index has gaps
    which breaks row-level lookups in the grader.
    """
    return [
        "df = df.drop_duplicates().reset_index(drop=True)",
    ]


def _chosen_whitespace(cols: list[str]) -> list[str]:
    """str.strip() removes BOTH leading AND trailing whitespace."""
    return [f"df['{col}'] = df['{col}'].str.strip()" for col in cols]


def _chosen_format(cols: list[str], clean_df: pd.DataFrame) -> list[str]:
    """Restore the original casing inferred from the clean data."""
    lines = []
    for col in cols:
        case = _detect_case(clean_df, col)
        if case == "upper":
            lines.append(f"df['{col}'] = df['{col}'].str.upper()")
        elif case == "lower":
            lines.append(f"df['{col}'] = df['{col}'].str.lower()")
        else:
            lines.append(f"df['{col}'] = df['{col}'].str.title()")
    return lines


def _chosen_outliers(cols: list[str], clean_df: pd.DataFrame) -> list[str]:
    """
    Clip outliers to the clean data's actual min/max — the safest correct bound.
    We round to 2 decimal places to keep the code readable.
    """
    lines = []
    for col in cols:
        lo, hi = _clean_col_bounds(clean_df, col)
        lines.append(
            f"df['{col}'] = pd.to_numeric(df['{col}'], errors='coerce')"
            f".clip(lower={lo:.2f}, upper={hi:.2f})"
        )
    return lines


# ── Rejected fix templates (deliberate mistakes) ──────────────────────────────


def _rejected_nulls(cols: list[str], clean_df: pd.DataFrame) -> list[str]:
    """
    MISTAKE: fillna(0) for numeric, fillna('unknown') for categorical.
    This is a very common beginner mistake — it "fixes" the null technically
    but puts a wrong value in (0 for Age is meaningless, not a real age).
    The grader will penalise this because _is_reasonable_fill() will reject 0
    for columns where 0 is outside the valid range.

    Extra mistake: only fix the FIRST column, skip the rest.
    Missing a column is another common real-world error.
    """
    lines = []
    targets = cols[:1]          # deliberately only fix first column
    for col in targets:
        if _is_numeric(clean_df, col):
            lines.append(f"df['{col}'] = df['{col}'].fillna(0)")
        else:
            lines.append(f"df['{col}'] = df['{col}'].fillna('unknown')")
    return lines


def _rejected_type_mangle(cols: list[str], clean_df: pd.DataFrame) -> list[str]:
    """
    MISTAKE: convert to numeric but forget to fillna afterward.
    pd.to_numeric(..., errors='coerce') turns garbage → NaN,
    but if you stop there the column still has nulls — just a different kind.
    The grader will penalise because the null_injected errors aren't fixed.
    """
    lines = []
    for col in cols:
        lines.append(f"df['{col}'] = pd.to_numeric(df['{col}'], errors='coerce')")
        # Intentionally omitting the fillna — this is the mistake
    return lines


def _rejected_duplicates() -> list[str]:
    """
    MISTAKE: drop duplicates but forget reset_index.
    The index will have gaps (e.g., [0,1,3,5,6...]) and the grader's
    row-level lookups will fail to find rows by integer position.
    """
    return [
        "df = df.drop_duplicates()",   # no reset_index — intentional mistake
    ]


def _rejected_whitespace(cols: list[str]) -> list[str]:
    """
    MISTAKE: lstrip() only removes LEADING spaces, misses trailing spaces.
    The grader checks str.strip() == str, so trailing spaces still fail.
    """
    return [f"df['{col}'] = df['{col}'].str.lstrip()" for col in cols]


def _rejected_format(cols: list[str], clean_df: pd.DataFrame) -> list[str]:
    """
    MISTAKE: always lowercase regardless of the actual intended case.
    If the clean data uses UPPER (like 'S', 'C', 'Q' for Embarked),
    lowercasing them makes them wrong values.
    """
    return [f"df['{col}'] = df['{col}'].str.lower()" for col in cols]


def _rejected_outliers(cols: list[str], clean_df: pd.DataFrame) -> list[str]:
    """
    MISTAKE: only clip the lower bound, ignore the upper bound.
    Upper outliers (e.g., Age=847) are the main problem but go unfixed.
    """
    lines = []
    for col in cols:
        lo, _ = _clean_col_bounds(clean_df, col)
        lines.append(
            f"df['{col}'] = pd.to_numeric(df['{col}'], errors='coerce')"
            f".clip(lower={lo:.2f})"   # no upper — intentional mistake
        )
    return lines


# ── Script builder ────────────────────────────────────────────────────────────


def _build_script(lines: list[str]) -> str:
    """
    Wrap fix lines in a complete runnable Python script.
    The environment's sandbox expects `df` to already be loaded,
    so we just add the import and let the sandbox preamble handle df loading.
    """
    if not lines:
        return "# no fixes applied"
    header = "import pandas as pd\nimport numpy as np\n"
    return header + "\n".join(lines)


# ── Public API ────────────────────────────────────────────────────────────────


def synthesize_chosen(error_map: dict[str, Any], clean_df: pd.DataFrame) -> str:
    """
    Generate the correct pandas fix code from the mutation map.

    Reads which cells broke and with what corruption type, then applies
    the appropriate fix template for each group of columns.

    Returns a complete runnable Python script string.
    """
    groups = _group_errors(error_map)
    lines: list[str] = []

    # Process in a fixed order so the output is deterministic and readable
    if "duplicate_rows" in groups:
        # Always drop duplicates first — removes spurious rows before other fixes
        lines += _chosen_duplicates()

    if "type_mangled" in groups:
        lines += _chosen_type_mangle(groups["type_mangled"], clean_df)

    if "null_injected" in groups:
        lines += _chosen_nulls(groups["null_injected"], clean_df)

    if "outlier_injected" in groups:
        lines += _chosen_outliers(groups["outlier_injected"], clean_df)

    if "whitespace_noise" in groups:
        lines += _chosen_whitespace(groups["whitespace_noise"])

    if "format_inconsistency" in groups:
        lines += _chosen_format(groups["format_inconsistency"], clean_df)

    return _build_script(lines)


def synthesize_rejected(error_map: dict[str, Any], clean_df: pd.DataFrame) -> str:
    """
    Generate plausible-but-wrong pandas code from the mutation map.

    Uses the same structure as synthesize_chosen but applies deliberate mistakes
    for each corruption type. These are realistic mistakes — the kind an LLM
    or junior analyst might actually make.

    Returns a complete runnable Python script string.
    """
    groups = _group_errors(error_map)
    lines: list[str] = []

    if "duplicate_rows" in groups:
        lines += _rejected_duplicates()         # forgets reset_index

    if "type_mangled" in groups:
        lines += _rejected_type_mangle(groups["type_mangled"], clean_df)  # forgets fillna

    if "null_injected" in groups:
        lines += _rejected_nulls(groups["null_injected"], clean_df)       # fillna(0), misses cols

    if "outlier_injected" in groups:
        lines += _rejected_outliers(groups["outlier_injected"], clean_df) # no upper bound

    if "whitespace_noise" in groups:
        lines += _rejected_whitespace(groups["whitespace_noise"])         # lstrip only

    if "format_inconsistency" in groups:
        lines += _rejected_format(groups["format_inconsistency"], clean_df)  # wrong case

    return _build_script(lines)
