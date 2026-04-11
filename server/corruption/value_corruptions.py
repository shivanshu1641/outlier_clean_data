"""All 22 value-level corruption functions for the runtime pipeline.

Every function follows this signature:
    def corruption_name(df, columns, fraction, error_log, clean_df, rng, py_rng, **kwargs) -> pd.DataFrame

Each function:
1. Works on a copy of df
2. Applies corruption to ``fraction`` of rows in specified columns
3. Appends entries to ``error_log`` for every corrupted cell
4. Returns the corrupted DataFrame
5. Is deterministic given the same RNG state
"""
from __future__ import annotations

import html
import math
import random
import string
from datetime import datetime as _dt
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Severity per corruption type
# ---------------------------------------------------------------------------

CORRUPTION_SEVERITY: dict[str, float] = {
    "inject_nulls": 2.0,
    "type_mangle": 3.0,
    "duplicate_rows": 2.0,
    "whitespace_noise": 1.0,
    "format_inconsistency": 1.0,
    "outlier_injection": 3.0,
    "drop_rows": 2.5,
    "decimal_shift": 3.0,
    "value_swap": 2.5,
    "typo_injection": 1.5,
    "date_format_mix": 1.5,
    "abbreviation_mix": 1.5,
    "leading_zero_strip": 2.0,
    "header_in_data": 2.0,
    "category_misspell": 1.5,
    "business_rule_violation": 3.0,
    "encoding_noise": 2.5,
    "schema_drift": 3.0,
    "unicode_homoglyph": 2.5,
    "html_entity_leak": 1.5,
    "column_shift": 3.0,
    "unit_inconsistency": 2.5,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_clean_val(clean_df: pd.DataFrame | None, df: pd.DataFrame, idx: int, col: str) -> Any:
    """Get clean value from original clean_df if available and in bounds, else from current df."""
    if clean_df is not None and idx < len(clean_df) and col in clean_df.columns:
        return clean_df.at[idx, col]
    return df.at[idx, col]


def _safe_clean_value(val: Any) -> Any:
    """Normalise a clean value for the error log (avoid numpy types)."""
    if val is None:
        return None
    try:
        if pd.isna(val):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    return val


def _choose_indices(rng: np.random.Generator, n_total: int, fraction: float) -> np.ndarray:
    """Select deterministic row positions to corrupt."""
    n_corrupt = max(1, int(n_total * fraction))
    n_corrupt = min(n_corrupt, n_total)
    return rng.choice(n_total, size=n_corrupt, replace=False)


# ===================================================================
# 1. inject_nulls
# ===================================================================

def inject_nulls(
    df: pd.DataFrame,
    columns: list[str],
    fraction: float = 0.1,
    error_log: list[dict] | None = None,
    clean_df: pd.DataFrame | None = None,
    rng: np.random.Generator | None = None,
    py_rng: random.Random | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Randomly set cells to NaN."""
    df = df.copy()
    rng = rng or np.random.default_rng(42)
    severity = CORRUPTION_SEVERITY["inject_nulls"]
    for col in columns:
        if col not in df.columns:
            continue
        mask = rng.random(len(df)) < fraction
        indices = df.index[mask]
        if error_log is not None:
            for idx in indices:
                val = _get_clean_val(clean_df, df, idx, col)
                if not pd.isna(val):
                    error_log.append({
                        "key": f"{idx},{col}",
                        "severity": severity,
                        "clean_value": _safe_clean_value(val),
                        "corruption": "inject_nulls",
                        "accepted_fill": "any",
                    })
        df.loc[indices, col] = np.nan
    return df


# ===================================================================
# 2. type_mangle
# ===================================================================

def type_mangle(
    df: pd.DataFrame,
    columns: list[str],
    fraction: float = 0.05,
    error_log: list[dict] | None = None,
    clean_df: pd.DataFrame | None = None,
    rng: np.random.Generator | None = None,
    py_rng: random.Random | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Mix string garbage into numeric columns."""
    df = df.copy()
    rng = rng or np.random.default_rng(42)
    severity = CORRUPTION_SEVERITY["type_mangle"]
    garbage = ["N/A", "unknown", "##", "n/a", "-", "???", "null", "NA"]
    for col in columns:
        if col not in df.columns:
            continue
        df[col] = df[col].astype(object)
        idxs = _choose_indices(rng, len(df), fraction)
        for pos in idxs:
            idx = df.index[pos]
            clean_val = _get_clean_val(clean_df, df, idx, col)
            garb = garbage[int(rng.integers(0, len(garbage)))]
            if error_log is not None:
                error_log.append({
                    "key": f"{idx},{col}",
                    "severity": severity,
                    "clean_value": _safe_clean_value(clean_val),
                    "corruption": "type_mangle",
                })
            df.at[idx, col] = garb
    return df


# ===================================================================
# 3. duplicate_rows
# ===================================================================

def duplicate_rows(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    fraction: float = 0.05,
    error_log: list[dict] | None = None,
    clean_df: pd.DataFrame | None = None,
    rng: np.random.Generator | None = None,
    py_rng: random.Random | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Insert exact duplicate rows."""
    rng = rng or np.random.default_rng(42)
    n_dupes = max(1, int(len(df) * fraction))
    severity = CORRUPTION_SEVERITY["duplicate_rows"]
    dupe_pos = rng.choice(len(df), size=n_dupes, replace=True)
    dupes = df.iloc[dupe_pos].copy()
    result = pd.concat([df, dupes], ignore_index=True)
    if error_log is not None:
        start_idx = len(df)
        for i in range(n_dupes):
            error_log.append({
                "key": f"spurious_{start_idx + i}",
                "severity": severity,
                "clean_value": None,
                "corruption": "duplicate_rows",
            })
    return result


# ===================================================================
# 4. whitespace_noise
# ===================================================================

def whitespace_noise(
    df: pd.DataFrame,
    columns: list[str],
    fraction: float = 0.1,
    error_log: list[dict] | None = None,
    clean_df: pd.DataFrame | None = None,
    rng: np.random.Generator | None = None,
    py_rng: random.Random | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Add leading/trailing/double spaces."""
    df = df.copy()
    rng = rng or np.random.default_rng(42)
    severity = CORRUPTION_SEVERITY["whitespace_noise"]
    for col in columns:
        if col not in df.columns or df[col].dtype != object:
            continue
        idxs = _choose_indices(rng, len(df), fraction)
        for pos in idxs:
            idx = df.index[pos]
            val = str(df.at[idx, col])
            clean_val = _get_clean_val(clean_df, df, idx, col)
            choice = int(rng.integers(0, 3))
            if choice == 0:
                new_val = "  " + val
            elif choice == 1:
                new_val = val + "  "
            else:
                new_val = val.replace(" ", "  ", 1) if " " in val else "  " + val
            if error_log is not None:
                error_log.append({
                    "key": f"{idx},{col}",
                    "severity": severity,
                    "clean_value": _safe_clean_value(clean_val),
                    "corruption": "whitespace_noise",
                })
            df.at[idx, col] = new_val
    return df


# ===================================================================
# 5. format_inconsistency
# ===================================================================

def format_inconsistency(
    df: pd.DataFrame,
    columns: list[str],
    fraction: float = 0.15,
    error_log: list[dict] | None = None,
    clean_df: pd.DataFrame | None = None,
    rng: np.random.Generator | None = None,
    py_rng: random.Random | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Inject inconsistent string formats (casing changes)."""
    df = df.copy()
    rng = rng or np.random.default_rng(42)
    severity = CORRUPTION_SEVERITY["format_inconsistency"]
    for col in columns:
        if col not in df.columns or df[col].dtype != object:
            continue
        idxs = _choose_indices(rng, len(df), fraction)
        for pos in idxs:
            idx = df.index[pos]
            val = str(df.at[idx, col])
            clean_val = _get_clean_val(clean_df, df, idx, col)
            choice = int(rng.integers(0, 3))
            new_val = val.upper() if choice == 0 else (val.lower() if choice == 1 else val.title())
            if new_val != val:
                df.at[idx, col] = new_val
                if error_log is not None:
                    error_log.append({
                        "key": f"{idx},{col}",
                        "severity": severity,
                        "clean_value": _safe_clean_value(clean_val),
                        "corruption": "format_inconsistency",
                    })
    return df


# ===================================================================
# 6. outlier_injection
# ===================================================================

def outlier_injection(
    df: pd.DataFrame,
    columns: list[str],
    fraction: float = 0.02,
    error_log: list[dict] | None = None,
    clean_df: pd.DataFrame | None = None,
    rng: np.random.Generator | None = None,
    py_rng: random.Random | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Inject extreme outlier values into numeric columns."""
    df = df.copy()
    rng = rng or np.random.default_rng(42)
    severity = CORRUPTION_SEVERITY["outlier_injection"]
    for col in columns:
        if col not in df.columns:
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.notna().sum() == 0:
            continue
        if pd.api.types.is_integer_dtype(df[col]):
            df[col] = df[col].astype(float)
        col_mean = numeric.mean()
        col_std = numeric.std()
        if col_std == 0:
            col_std = 1.0
        idxs = _choose_indices(rng, len(df), fraction)
        for pos in idxs:
            idx = df.index[pos]
            clean_val = _get_clean_val(clean_df, df, idx, col)
            multiplier = float(rng.uniform(5, 20))
            new_val = col_mean + col_std * multiplier if rng.random() > 0.5 else col_mean - col_std * multiplier
            if error_log is not None:
                error_log.append({
                    "key": f"{idx},{col}",
                    "severity": severity,
                    "clean_value": _safe_clean_value(clean_val),
                    "corruption": "outlier_injection",
                })
            df.at[idx, col] = new_val
    return df


# ===================================================================
# 7. drop_rows
# ===================================================================

def drop_rows(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    fraction: float = 0.03,
    error_log: list[dict] | None = None,
    clean_df: pd.DataFrame | None = None,
    rng: np.random.Generator | None = None,
    py_rng: random.Random | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Randomly remove rows from the DataFrame."""
    rng = rng or np.random.default_rng(42)
    n_drop = max(1, int(len(df) * fraction))
    severity = CORRUPTION_SEVERITY["drop_rows"]
    drop_positions = rng.choice(len(df), size=n_drop, replace=False)
    drop_indices = df.index[drop_positions]
    if error_log is not None:
        for idx in drop_indices:
            row_values = {}
            if clean_df is not None and idx < len(clean_df):
                row_values = clean_df.iloc[idx].to_dict()
            error_log.append({
                "key": f"missing_{idx}",
                "severity": severity,
                "clean_values": {k: _safe_clean_value(v) for k, v in row_values.items()},
                "corruption": "drop_rows",
            })
    return df.drop(index=drop_indices).reset_index(drop=True)


# ===================================================================
# 8. decimal_shift
# ===================================================================

def decimal_shift(
    df: pd.DataFrame,
    columns: list[str],
    fraction: float = 0.03,
    error_log: list[dict] | None = None,
    clean_df: pd.DataFrame | None = None,
    rng: np.random.Generator | None = None,
    py_rng: random.Random | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Shift decimal point by multiplying/dividing by 10 or 100."""
    df = df.copy()
    rng = rng or np.random.default_rng(42)
    severity = CORRUPTION_SEVERITY["decimal_shift"]
    shifts = [0.01, 0.1, 10, 100]
    for col in columns:
        if col not in df.columns:
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.notna().sum() == 0:
            continue
        # Convert int columns to float so shifted values can be stored
        if pd.api.types.is_integer_dtype(df[col]):
            df[col] = df[col].astype(float)
        idxs = _choose_indices(rng, len(df), fraction)
        for pos in idxs:
            idx = df.index[pos]
            clean_val = _get_clean_val(clean_df, df, idx, col)
            try:
                fval = float(df.at[idx, col])
            except (TypeError, ValueError):
                continue
            if math.isnan(fval) or math.isinf(fval):
                continue
            shift = shifts[int(rng.integers(0, len(shifts)))]
            df.at[idx, col] = fval * shift
            if error_log is not None:
                error_log.append({
                    "key": f"{idx},{col}",
                    "severity": severity,
                    "clean_value": _safe_clean_value(clean_val),
                    "corruption": "decimal_shift",
                })
    return df


# ===================================================================
# 9. value_swap
# ===================================================================

def value_swap(
    df: pd.DataFrame,
    columns: list[str],
    fraction: float = 0.03,
    error_log: list[dict] | None = None,
    clean_df: pd.DataFrame | None = None,
    rng: np.random.Generator | None = None,
    py_rng: random.Random | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Swap values between pairs of columns for some rows."""
    df = df.copy()
    rng = rng or np.random.default_rng(42)
    severity = CORRUPTION_SEVERITY["value_swap"]
    if len(columns) < 2:
        return df
    for i in range(0, len(columns) - 1, 2):
        col_a, col_b = columns[i], columns[i + 1]
        if col_a not in df.columns or col_b not in df.columns:
            continue
        idxs = _choose_indices(rng, len(df), fraction)
        for pos in idxs:
            idx = df.index[pos]
            val_a, val_b = df.at[idx, col_a], df.at[idx, col_b]
            clean_a = _get_clean_val(clean_df, df, idx, col_a)
            clean_b = _get_clean_val(clean_df, df, idx, col_b)
            df.at[idx, col_a], df.at[idx, col_b] = val_b, val_a
            if error_log is not None:
                error_log.append({
                    "key": f"{idx},{col_a}",
                    "severity": severity,
                    "clean_value": _safe_clean_value(clean_a),
                    "corruption": "value_swap",
                })
                error_log.append({
                    "key": f"{idx},{col_b}",
                    "severity": severity,
                    "clean_value": _safe_clean_value(clean_b),
                    "corruption": "value_swap",
                })
    return df


# ===================================================================
# 10. typo_injection
# ===================================================================

def typo_injection(
    df: pd.DataFrame,
    columns: list[str],
    fraction: float = 0.05,
    error_log: list[dict] | None = None,
    clean_df: pd.DataFrame | None = None,
    rng: np.random.Generator | None = None,
    py_rng: random.Random | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Inject character-level typos: transpose, delete, substitute."""
    df = df.copy()
    rng = rng or np.random.default_rng(42)
    py_rng = py_rng or random.Random(42)
    severity = CORRUPTION_SEVERITY["typo_injection"]
    for col in columns:
        if col not in df.columns or df[col].dtype != object:
            continue
        idxs = _choose_indices(rng, len(df), fraction)
        for pos in idxs:
            idx = df.index[pos]
            val = str(df.at[idx, col])
            if len(val) < 3:
                continue
            clean_val = _get_clean_val(clean_df, df, idx, col)
            op = int(rng.integers(0, 3))
            chars = list(val)
            if op == 0 and len(chars) >= 2:
                p = py_rng.randint(0, len(chars) - 2)
                chars[p], chars[p + 1] = chars[p + 1], chars[p]
            elif op == 1:
                p = py_rng.randint(0, len(chars) - 1)
                chars.pop(p)
            else:
                p = py_rng.randint(0, len(chars) - 1)
                chars[p] = py_rng.choice(string.ascii_lowercase)
            new_val = "".join(chars)
            if new_val != val:
                df.at[idx, col] = new_val
                if error_log is not None:
                    error_log.append({
                        "key": f"{idx},{col}",
                        "severity": severity,
                        "clean_value": _safe_clean_value(clean_val),
                        "corruption": "typo_injection",
                    })
    return df


# ===================================================================
# 11. date_format_mix  (NEW)
# ===================================================================

_DATE_FORMATS = [
    "%Y-%m-%d",       # 2024-01-15
    "%m/%d/%Y",       # 01/15/2024
    "%b %d %Y",       # Jan 15 2024
    "%d-%b-%Y",       # 15-Jan-2024
    "%Y/%m/%d",       # 2024/01/15
]


def date_format_mix(
    df: pd.DataFrame,
    columns: list[str],
    fraction: float = 0.1,
    error_log: list[dict] | None = None,
    clean_df: pd.DataFrame | None = None,
    rng: np.random.Generator | None = None,
    py_rng: random.Random | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Mix date formats within a column (2024-01-15, 01/15/2024, Jan 15 2024, etc.)."""
    df = df.copy()
    rng = rng or np.random.default_rng(42)
    severity = CORRUPTION_SEVERITY["date_format_mix"]
    for col in columns:
        if col not in df.columns or df[col].dtype != object:
            continue
        idxs = _choose_indices(rng, len(df), fraction)
        for pos in idxs:
            idx = df.index[pos]
            val = str(df.at[idx, col])
            clean_val = _get_clean_val(clean_df, df, idx, col)
            # Try parsing the value as a date using known formats first
            parsed = None
            for fmt in _DATE_FORMATS:
                try:
                    parsed = pd.Timestamp(_dt.strptime(val, fmt))
                    break
                except (ValueError, TypeError):
                    continue
            if parsed is None:
                try:
                    parsed = pd.to_datetime(val, format="mixed")
                except (ValueError, TypeError):
                    # Last resort: try pd.Timestamp directly
                    try:
                        parsed = pd.Timestamp(val)
                    except (ValueError, TypeError):
                        continue
            if parsed is pd.NaT:
                continue
            # Pick a different format
            new_fmt = _DATE_FORMATS[int(rng.integers(0, len(_DATE_FORMATS)))]
            new_val = parsed.strftime(new_fmt)
            if new_val != val:
                df.at[idx, col] = new_val
                if error_log is not None:
                    error_log.append({
                        "key": f"{idx},{col}",
                        "severity": severity,
                        "clean_value": _safe_clean_value(clean_val),
                        "corruption": "date_format_mix",
                    })
    return df


# ===================================================================
# 12. abbreviation_mix  (NEW)
# ===================================================================

_ABBREV_MAP: dict[str, list[str]] = {
    "california": ["CA", "Calif.", "Cali"],
    "new york": ["NY", "N.Y.", "NewYork"],
    "texas": ["TX", "Tex."],
    "florida": ["FL", "Fla."],
    "illinois": ["IL", "Ill."],
    "pennsylvania": ["PA", "Penn.", "Penna."],
    "ohio": ["OH"],
    "georgia": ["GA", "Ga."],
    "michigan": ["MI", "Mich."],
    "north carolina": ["NC", "N.C."],
    "street": ["St.", "St", "Str."],
    "avenue": ["Ave.", "Ave", "Av."],
    "boulevard": ["Blvd.", "Blvd"],
    "doctor": ["Dr.", "Dr"],
    "mister": ["Mr.", "Mr"],
    "junior": ["Jr.", "Jr"],
    "senior": ["Sr.", "Sr"],
    "united states": ["US", "U.S.", "USA", "U.S.A."],
    "company": ["Co.", "Co"],
    "corporation": ["Corp.", "Corp"],
    "incorporated": ["Inc.", "Inc"],
    "department": ["Dept.", "Dept"],
}


def abbreviation_mix(
    df: pd.DataFrame,
    columns: list[str],
    fraction: float = 0.1,
    error_log: list[dict] | None = None,
    clean_df: pd.DataFrame | None = None,
    rng: np.random.Generator | None = None,
    py_rng: random.Random | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Mix abbreviations (California/CA/Calif.)."""
    df = df.copy()
    rng = rng or np.random.default_rng(42)
    py_rng = py_rng or random.Random(42)
    severity = CORRUPTION_SEVERITY["abbreviation_mix"]
    for col in columns:
        if col not in df.columns or df[col].dtype != object:
            continue
        idxs = _choose_indices(rng, len(df), fraction)
        for pos in idxs:
            idx = df.index[pos]
            val = str(df.at[idx, col])
            clean_val = _get_clean_val(clean_df, df, idx, col)
            lower_val = val.lower()
            replaced = False
            # Check if the value matches any full form or abbreviation
            for full_form, abbrevs in _ABBREV_MAP.items():
                if lower_val == full_form or lower_val == full_form.title() or lower_val in [a.lower() for a in abbrevs]:
                    # Pick a random abbreviation that differs from current
                    candidates = [full_form.title()] + abbrevs
                    candidates = [c for c in candidates if c.lower() != lower_val]
                    if candidates:
                        new_val = py_rng.choice(candidates)
                        df.at[idx, col] = new_val
                        replaced = True
                        break
                # Also check if full_form appears as a substring
                if full_form in lower_val:
                    replacement = py_rng.choice(abbrevs)
                    new_val = val.lower().replace(full_form, replacement, 1)
                    # Restore original casing style
                    if val[0].isupper():
                        new_val = new_val[0].upper() + new_val[1:]
                    if new_val != val:
                        df.at[idx, col] = new_val
                        replaced = True
                        break
            if replaced and error_log is not None:
                error_log.append({
                    "key": f"{idx},{col}",
                    "severity": severity,
                    "clean_value": _safe_clean_value(clean_val),
                    "corruption": "abbreviation_mix",
                })
    return df


# ===================================================================
# 13. leading_zero_strip  (NEW)
# ===================================================================

def leading_zero_strip(
    df: pd.DataFrame,
    columns: list[str],
    fraction: float = 0.1,
    error_log: list[dict] | None = None,
    clean_df: pd.DataFrame | None = None,
    rng: np.random.Generator | None = None,
    py_rng: random.Random | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Remove leading zeros from string values (02101 -> 2101, 007 -> 7)."""
    df = df.copy()
    rng = rng or np.random.default_rng(42)
    severity = CORRUPTION_SEVERITY["leading_zero_strip"]
    for col in columns:
        if col not in df.columns or df[col].dtype != object:
            continue
        idxs = _choose_indices(rng, len(df), fraction)
        for pos in idxs:
            idx = df.index[pos]
            val = str(df.at[idx, col])
            clean_val = _get_clean_val(clean_df, df, idx, col)
            # Only apply if the value starts with '0' and has digits
            if len(val) > 1 and val[0] == "0" and val.isdigit():
                new_val = val.lstrip("0") or "0"
                if new_val != val:
                    df.at[idx, col] = new_val
                    if error_log is not None:
                        error_log.append({
                            "key": f"{idx},{col}",
                            "severity": severity,
                            "clean_value": _safe_clean_value(clean_val),
                            "corruption": "leading_zero_strip",
                        })
            else:
                # If no leading zeros, add leading zeros (reverse corruption)
                if val.isdigit() and len(val) >= 1:
                    n_zeros = int(rng.integers(1, 4))
                    new_val = "0" * n_zeros + val
                    df.at[idx, col] = new_val
                    if error_log is not None:
                        error_log.append({
                            "key": f"{idx},{col}",
                            "severity": severity,
                            "clean_value": _safe_clean_value(clean_val),
                            "corruption": "leading_zero_strip",
                        })
    return df


# ===================================================================
# 14. header_in_data  (NEW)
# ===================================================================

def header_in_data(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    fraction: float = 0.02,
    error_log: list[dict] | None = None,
    clean_df: pd.DataFrame | None = None,
    rng: np.random.Generator | None = None,
    py_rng: random.Random | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Insert the header row as a data row at one or more random positions."""
    df = df.copy()
    rng = rng or np.random.default_rng(42)
    severity = CORRUPTION_SEVERITY["header_in_data"]
    n_inserts = max(1, int(len(df) * fraction))
    header_row = pd.DataFrame([df.columns.tolist()], columns=df.columns)
    # Cast all values to object to avoid type conflicts
    for c in df.columns:
        df[c] = df[c].astype(object)

    insert_positions = sorted(rng.choice(len(df), size=n_inserts, replace=True))
    pieces = []
    prev = 0
    inserted_count = 0
    for insert_pos in insert_positions:
        pieces.append(df.iloc[prev:insert_pos])
        pieces.append(header_row.copy())
        prev = insert_pos
        inserted_count += 1
    pieces.append(df.iloc[prev:])
    result = pd.concat(pieces, ignore_index=True)

    if error_log is not None:
        # The inserted header rows are spurious rows
        offset = 0
        for insert_pos in insert_positions:
            actual_idx = int(insert_pos) + offset
            error_log.append({
                "key": f"spurious_{actual_idx}",
                "severity": severity,
                "clean_value": None,
                "corruption": "header_in_data",
            })
            offset += 1
    return result


# ===================================================================
# 15. category_misspell  (NEW)
# ===================================================================

def category_misspell(
    df: pd.DataFrame,
    columns: list[str],
    fraction: float = 0.1,
    error_log: list[dict] | None = None,
    clean_df: pd.DataFrame | None = None,
    rng: np.random.Generator | None = None,
    py_rng: random.Random | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Misspell categorical values (Male -> Mal, Female -> Femal, Yes -> Yse)."""
    df = df.copy()
    rng = rng or np.random.default_rng(42)
    py_rng = py_rng or random.Random(42)
    severity = CORRUPTION_SEVERITY["category_misspell"]
    for col in columns:
        if col not in df.columns or df[col].dtype != object:
            continue
        idxs = _choose_indices(rng, len(df), fraction)
        for pos in idxs:
            idx = df.index[pos]
            val = str(df.at[idx, col])
            if len(val) < 2:
                continue
            clean_val = _get_clean_val(clean_df, df, idx, col)
            chars = list(val)
            # Apply one of: truncate end, swap adjacent, double letter, drop letter
            op = int(rng.integers(0, 4))
            if op == 0 and len(chars) > 2:
                # Truncate last 1-2 chars
                n_trunc = py_rng.randint(1, min(2, len(chars) - 1))
                new_val = val[:-n_trunc]
            elif op == 1 and len(chars) >= 2:
                # Swap two adjacent chars
                p = py_rng.randint(0, len(chars) - 2)
                chars[p], chars[p + 1] = chars[p + 1], chars[p]
                new_val = "".join(chars)
            elif op == 2:
                # Double a letter
                p = py_rng.randint(0, len(chars) - 1)
                new_val = val[:p] + val[p] + val[p:]
            else:
                # Drop a letter from the middle
                if len(chars) > 2:
                    p = py_rng.randint(1, len(chars) - 2)
                    new_val = val[:p] + val[p + 1:]
                else:
                    new_val = val[:-1]
            if new_val != val:
                df.at[idx, col] = new_val
                if error_log is not None:
                    error_log.append({
                        "key": f"{idx},{col}",
                        "severity": severity,
                        "clean_value": _safe_clean_value(clean_val),
                        "corruption": "category_misspell",
                    })
    return df


# ===================================================================
# 16. business_rule_violation  (NEW)
# ===================================================================

def business_rule_violation(
    df: pd.DataFrame,
    columns: list[str],
    fraction: float = 0.05,
    error_log: list[dict] | None = None,
    clean_df: pd.DataFrame | None = None,
    rng: np.random.Generator | None = None,
    py_rng: random.Random | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Inject values that violate semantic rules.

    If ``rules`` is provided via kwargs, uses them to create targeted
    violations. Otherwise falls back to generic impossible-value injection.
    """
    df = df.copy()
    rng = rng or np.random.default_rng(42)
    py_rng = py_rng or random.Random(42)
    severity = CORRUPTION_SEVERITY["business_rule_violation"]
    rules = kwargs.get("rules", [])

    if rules:
        return _rule_aware_violation(df, rules, fraction, error_log, clean_df, rng, py_rng, severity)

    # Fallback: generic impossible-value injection (existing behavior)
    for col in columns:
        if col not in df.columns:
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.notna().sum() == 0:
            continue
        if pd.api.types.is_integer_dtype(df[col]):
            df[col] = df[col].astype(float)
        col_min = numeric.min()
        col_max = numeric.max()
        col_range = col_max - col_min if col_max != col_min else abs(col_max) or 1.0
        idxs = _choose_indices(rng, len(df), fraction)
        for pos in idxs:
            idx = df.index[pos]
            clean_val = _get_clean_val(clean_df, df, idx, col)
            try:
                fval = float(df.at[idx, col])
            except (TypeError, ValueError):
                continue
            if math.isnan(fval) or math.isinf(fval):
                continue
            violation_type = int(rng.integers(0, 3))
            if violation_type == 0:
                new_val = -abs(fval) - float(rng.uniform(1, col_range * 0.5))
            elif violation_type == 1:
                new_val = col_max + col_range * float(rng.uniform(2, 10))
            else:
                new_val = -fval if fval > 0 else fval - float(rng.uniform(1, col_range))
            df.at[idx, col] = new_val
            if error_log is not None:
                error_log.append({
                    "key": f"{idx},{col}",
                    "severity": severity,
                    "clean_value": _safe_clean_value(clean_val),
                    "corruption": "business_rule_violation",
                })
    return df


def _rule_aware_violation(
    df: pd.DataFrame,
    rules: list,
    fraction: float,
    error_log: list[dict] | None,
    clean_df: pd.DataFrame | None,
    rng: np.random.Generator,
    py_rng: random.Random,
    severity: float,
) -> pd.DataFrame:
    """Create targeted violations based on semantic rules."""
    try:
        from server.rules.types import RangeRule, RegexRule, EnumRule, NotNullRule, UniqueRule
    except ImportError:
        from rules.types import RangeRule, RegexRule, EnumRule, NotNullRule, UniqueRule

    n_rows = len(df)
    n_corrupt = max(1, int(n_rows * fraction))
    rows = rng.choice(n_rows, size=min(n_corrupt, n_rows), replace=False)

    applicable = [r for r in rules if hasattr(r, "column") and r.column in df.columns]
    if not applicable:
        return df

    for idx in rows:
        rule = py_rng.choice(applicable)
        col = rule.column
        clean_val = _get_clean_val(clean_df, df, int(idx), col)

        if isinstance(rule, RangeRule):
            if rng.random() < 0.5 and rule.min_val != float("-inf"):
                spread = abs(rule.max_val - rule.min_val) * 0.5 + 1 if rule.max_val != float("inf") else 100
                bad_val = rule.min_val - float(rng.uniform(1, spread))
            else:
                if rule.max_val == float("inf"):
                    bad_val = -float(rng.uniform(1, 1000))
                else:
                    spread = abs(rule.max_val - rule.min_val) * 0.5 + 1
                    bad_val = rule.max_val + float(rng.uniform(1, spread))
            df.at[int(idx), col] = bad_val
        elif isinstance(rule, EnumRule):
            bad_val = f"INVALID_{int(rng.integers(100, 999))}"
            df.at[int(idx), col] = bad_val
        elif isinstance(rule, NotNullRule):
            df.at[int(idx), col] = None
        elif isinstance(rule, RegexRule):
            df.at[int(idx), col] = f"!!!INVALID_{int(rng.integers(100))}"
        elif isinstance(rule, UniqueRule):
            other_idx = int(rng.choice(n_rows))
            df.at[int(idx), col] = df.at[other_idx, col]
        else:
            continue

        if error_log is not None:
            error_log.append({
                "key": f"{int(idx)},{col}",
                "severity": severity,
                "clean_value": _safe_clean_value(clean_val),
                "corruption": "business_rule_violation",
                "rule_type": rule.rule_type,
            })

    return df


# ===================================================================
# 17. encoding_noise  (NEW)
# ===================================================================

_MOJIBAKE_MAP: dict[str, str] = {
    "\u00e9": "\u00c3\u00a9",   # e -> Ã©
    "\u00e8": "\u00c3\u00a8",   # e -> Ã¨
    "\u00e0": "\u00c3\u00a0",   # a -> Ã
    "\u00fc": "\u00c3\u00bc",   # u -> Ã¼
    "\u00f1": "\u00c3\u00b1",   # n -> Ã±
    "\u00e4": "\u00c3\u00a4",   # a -> Ã¤
    "\u00f6": "\u00c3\u00b6",   # o -> Ã¶
    "\u00df": "\u00c3\u009f",   # ss -> Ã
}

# Reverse map for injecting encoding issues into ASCII text
_ASCII_TO_ACCENTED: dict[str, str] = {
    "e": "\u00e9",
    "a": "\u00e0",
    "u": "\u00fc",
    "n": "\u00f1",
    "o": "\u00f6",
}


def encoding_noise(
    df: pd.DataFrame,
    columns: list[str],
    fraction: float = 0.05,
    error_log: list[dict] | None = None,
    clean_df: pd.DataFrame | None = None,
    rng: np.random.Generator | None = None,
    py_rng: random.Random | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Introduce mojibake-style encoding corruption into string values."""
    df = df.copy()
    rng = rng or np.random.default_rng(42)
    py_rng = py_rng or random.Random(42)
    severity = CORRUPTION_SEVERITY["encoding_noise"]
    for col in columns:
        if col not in df.columns or df[col].dtype != object:
            continue
        idxs = _choose_indices(rng, len(df), fraction)
        for pos in idxs:
            idx = df.index[pos]
            val = str(df.at[idx, col])
            clean_val = _get_clean_val(clean_df, df, idx, col)
            new_val = val
            # First, try to apply mojibake to existing accented chars
            has_accent = False
            for orig, mojibake in _MOJIBAKE_MAP.items():
                if orig in new_val:
                    new_val = new_val.replace(orig, mojibake, 1)
                    has_accent = True
                    break
            # If no accented chars, introduce them by replacing ASCII vowels
            if not has_accent and len(val) >= 2:
                replaceable = [(i, c) for i, c in enumerate(val) if c.lower() in _ASCII_TO_ACCENTED]
                if replaceable:
                    target_idx, target_char = py_rng.choice(replaceable)
                    accented = _ASCII_TO_ACCENTED[target_char.lower()]
                    mojibake = _MOJIBAKE_MAP.get(accented, accented)
                    chars = list(new_val)
                    chars[target_idx] = mojibake
                    new_val = "".join(chars)
            if new_val != val:
                df.at[idx, col] = new_val
                if error_log is not None:
                    error_log.append({
                        "key": f"{idx},{col}",
                        "severity": severity,
                        "clean_value": _safe_clean_value(clean_val),
                        "corruption": "encoding_noise",
                    })
    return df


# ===================================================================
# 18. schema_drift  (NEW)
# ===================================================================

def schema_drift(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    fraction: float = 0.05,
    error_log: list[dict] | None = None,
    clean_df: pd.DataFrame | None = None,
    rng: np.random.Generator | None = None,
    py_rng: random.Random | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Rename columns mid-dataset by swapping column data for a subset of rows.

    Simulates schema drift by shifting column values for selected rows,
    as if the column order changed partway through data collection.
    """
    df = df.copy()
    rng = rng or np.random.default_rng(42)
    severity = CORRUPTION_SEVERITY["schema_drift"]
    all_cols = df.columns.tolist()
    if len(all_cols) < 2:
        return df
    # Pick two columns to "drift" (swap their values for some rows)
    col_indices = rng.choice(len(all_cols), size=2, replace=False)
    col_a = all_cols[int(col_indices[0])]
    col_b = all_cols[int(col_indices[1])]
    # Cast both to object to avoid type conflicts
    df[col_a] = df[col_a].astype(object)
    df[col_b] = df[col_b].astype(object)
    idxs = _choose_indices(rng, len(df), fraction)
    for pos in idxs:
        idx = df.index[pos]
        clean_val_a = _get_clean_val(clean_df, df, idx, col_a)
        clean_val_b = _get_clean_val(clean_df, df, idx, col_b)
        val_a, val_b = df.at[idx, col_a], df.at[idx, col_b]
        df.at[idx, col_a] = val_b
        df.at[idx, col_b] = val_a
        if error_log is not None:
            error_log.append({
                "key": f"{idx},{col_a}",
                "severity": severity,
                "clean_value": _safe_clean_value(clean_val_a),
                "corruption": "schema_drift",
            })
            error_log.append({
                "key": f"{idx},{col_b}",
                "severity": severity,
                "clean_value": _safe_clean_value(clean_val_b),
                "corruption": "schema_drift",
            })
    return df


# ===================================================================
# 19. unicode_homoglyph  (NEW)
# ===================================================================

# Map ASCII chars to Cyrillic/Greek look-alikes
_HOMOGLYPH_MAP: dict[str, str] = {
    "a": "\u0430",  # Cyrillic а
    "e": "\u0435",  # Cyrillic е
    "o": "\u043e",  # Cyrillic о
    "p": "\u0440",  # Cyrillic р
    "c": "\u0441",  # Cyrillic с
    "x": "\u0445",  # Cyrillic х
    "y": "\u0443",  # Cyrillic у
    "A": "\u0410",  # Cyrillic А
    "B": "\u0412",  # Cyrillic В
    "E": "\u0415",  # Cyrillic Е
    "H": "\u041d",  # Cyrillic Н
    "K": "\u041a",  # Cyrillic К
    "M": "\u041c",  # Cyrillic М
    "O": "\u041e",  # Cyrillic О
    "P": "\u0420",  # Cyrillic Р
    "T": "\u0422",  # Cyrillic Т
    "X": "\u0425",  # Cyrillic Х
}


def unicode_homoglyph(
    df: pd.DataFrame,
    columns: list[str],
    fraction: float = 0.05,
    error_log: list[dict] | None = None,
    clean_df: pd.DataFrame | None = None,
    rng: np.random.Generator | None = None,
    py_rng: random.Random | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Replace ASCII characters with visually similar Unicode homoglyphs."""
    df = df.copy()
    rng = rng or np.random.default_rng(42)
    py_rng = py_rng or random.Random(42)
    severity = CORRUPTION_SEVERITY["unicode_homoglyph"]
    for col in columns:
        if col not in df.columns or df[col].dtype != object:
            continue
        idxs = _choose_indices(rng, len(df), fraction)
        for pos in idxs:
            idx = df.index[pos]
            val = str(df.at[idx, col])
            clean_val = _get_clean_val(clean_df, df, idx, col)
            # Find replaceable characters
            replaceable = [(i, c) for i, c in enumerate(val) if c in _HOMOGLYPH_MAP]
            if not replaceable:
                continue
            # Replace 1-3 characters
            n_replace = min(int(rng.integers(1, 4)), len(replaceable))
            targets = py_rng.sample(replaceable, n_replace)
            chars = list(val)
            for char_idx, char in targets:
                chars[char_idx] = _HOMOGLYPH_MAP[char]
            new_val = "".join(chars)
            if new_val != val:
                df.at[idx, col] = new_val
                if error_log is not None:
                    error_log.append({
                        "key": f"{idx},{col}",
                        "severity": severity,
                        "clean_value": _safe_clean_value(clean_val),
                        "corruption": "unicode_homoglyph",
                    })
    return df


# ===================================================================
# 20. html_entity_leak  (NEW)
# ===================================================================

_HTML_ENTITY_MAP: dict[str, str] = {
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
    " ": "&nbsp;",
}


def html_entity_leak(
    df: pd.DataFrame,
    columns: list[str],
    fraction: float = 0.1,
    error_log: list[dict] | None = None,
    clean_df: pd.DataFrame | None = None,
    rng: np.random.Generator | None = None,
    py_rng: random.Random | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Replace characters with their HTML entity equivalents."""
    df = df.copy()
    rng = rng or np.random.default_rng(42)
    py_rng = py_rng or random.Random(42)
    severity = CORRUPTION_SEVERITY["html_entity_leak"]
    entity_chars = list(_HTML_ENTITY_MAP.keys())
    for col in columns:
        if col not in df.columns or df[col].dtype != object:
            continue
        idxs = _choose_indices(rng, len(df), fraction)
        for pos in idxs:
            idx = df.index[pos]
            val = str(df.at[idx, col])
            clean_val = _get_clean_val(clean_df, df, idx, col)
            new_val = val
            # Find characters that can be entity-encoded
            has_entity_char = any(c in val for c in entity_chars)
            if has_entity_char:
                # Replace one random entity-encodable character
                for c in entity_chars:
                    if c in new_val:
                        new_val = new_val.replace(c, _HTML_ENTITY_MAP[c], 1)
                        break
            else:
                # Escape the whole string via html.escape as fallback
                escaped = html.escape(val)
                if escaped != val:
                    new_val = escaped
                else:
                    # Wrap with a spurious entity
                    new_val = val + "&nbsp;"
            if new_val != val:
                df.at[idx, col] = new_val
                if error_log is not None:
                    error_log.append({
                        "key": f"{idx},{col}",
                        "severity": severity,
                        "clean_value": _safe_clean_value(clean_val),
                        "corruption": "html_entity_leak",
                    })
    return df


# ===================================================================
# 21. column_shift  (NEW)
# ===================================================================

def column_shift(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    fraction: float = 0.05,
    error_log: list[dict] | None = None,
    clean_df: pd.DataFrame | None = None,
    rng: np.random.Generator | None = None,
    py_rng: random.Random | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Circular-shift values within a column for a subset of rows.

    Shifts values down by a random offset within the selected row block,
    simulating a column alignment error.
    """
    df = df.copy()
    rng = rng or np.random.default_rng(42)
    severity = CORRUPTION_SEVERITY["column_shift"]
    cols_to_shift = columns if columns else df.columns.tolist()
    # Pick one column to shift
    if not cols_to_shift:
        return df
    target_col = cols_to_shift[int(rng.integers(0, len(cols_to_shift)))]
    if target_col not in df.columns:
        return df
    # Select a contiguous block of rows to shift
    n_rows = max(2, int(len(df) * fraction))
    n_rows = min(n_rows, len(df))
    start_pos = int(rng.integers(0, max(1, len(df) - n_rows)))
    end_pos = start_pos + n_rows
    # Circular shift by 1-3 positions
    shift_amount = int(rng.integers(1, min(4, n_rows)))
    block = df[target_col].iloc[start_pos:end_pos].values.copy()
    shifted_block = np.roll(block, shift_amount)
    # Log the errors before applying
    if error_log is not None:
        for i in range(start_pos, end_pos):
            idx = df.index[i]
            orig = block[i - start_pos]
            shifted = shifted_block[i - start_pos]
            try:
                same = orig == shifted
                if isinstance(same, np.ndarray):
                    same = same.all()
            except (ValueError, TypeError):
                same = False
            if not same:
                clean_val = _get_clean_val(clean_df, df, idx, target_col)
                error_log.append({
                    "key": f"{idx},{target_col}",
                    "severity": severity,
                    "clean_value": _safe_clean_value(clean_val),
                    "corruption": "column_shift",
                })
    # Apply the shift
    df.iloc[start_pos:end_pos, df.columns.get_loc(target_col)] = shifted_block
    return df


# ===================================================================
# 22. unit_inconsistency  (NEW)
# ===================================================================

_UNIT_CONVERSIONS: list[dict[str, Any]] = [
    {"factor": 2.20462, "from": "kg", "to": "lbs"},
    {"factor": 0.453592, "from": "lbs", "to": "kg"},
    {"factor": 3.28084, "from": "m", "to": "ft"},
    {"factor": 0.3048, "from": "ft", "to": "m"},
    {"factor": 1.60934, "from": "mi", "to": "km"},
    {"factor": 0.621371, "from": "km", "to": "mi"},
    {"factor": 33.814, "from": "L", "to": "fl oz"},
    {"factor": 0.0295735, "from": "fl oz", "to": "L"},
    {"factor": 1.8, "from": "C_offset", "to": "F_offset"},  # temp delta
]


def unit_inconsistency(
    df: pd.DataFrame,
    columns: list[str],
    fraction: float = 0.05,
    error_log: list[dict] | None = None,
    clean_df: pd.DataFrame | None = None,
    rng: np.random.Generator | None = None,
    py_rng: random.Random | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Mix units by applying random conversion factors to numeric values (kg/lbs, m/ft, etc.)."""
    df = df.copy()
    rng = rng or np.random.default_rng(42)
    severity = CORRUPTION_SEVERITY["unit_inconsistency"]
    for col in columns:
        if col not in df.columns:
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.notna().sum() == 0:
            continue
        if pd.api.types.is_integer_dtype(df[col]):
            df[col] = df[col].astype(float)
        # Pick a random conversion for this column
        conversion = _UNIT_CONVERSIONS[int(rng.integers(0, len(_UNIT_CONVERSIONS)))]
        factor = conversion["factor"]
        idxs = _choose_indices(rng, len(df), fraction)
        for pos in idxs:
            idx = df.index[pos]
            clean_val = _get_clean_val(clean_df, df, idx, col)
            try:
                fval = float(df.at[idx, col])
            except (TypeError, ValueError):
                continue
            if math.isnan(fval) or math.isinf(fval):
                continue
            new_val = fval * factor
            # Round to similar precision as original
            if fval != 0:
                decimals = max(0, -int(math.floor(math.log10(abs(fval)))) + 4)
                new_val = round(new_val, decimals)
            df.at[idx, col] = new_val
            if error_log is not None:
                error_log.append({
                    "key": f"{idx},{col}",
                    "severity": severity,
                    "clean_value": _safe_clean_value(clean_val),
                    "corruption": "unit_inconsistency",
                })
    return df


# ===================================================================
# Registry
# ===================================================================

CORRUPTION_REGISTRY: dict[str, dict[str, Any]] = {
    "inject_nulls": {"fn": inject_nulls, "requires_numeric": False, "requires_string": False},
    "type_mangle": {"fn": type_mangle, "requires_numeric": True, "requires_string": False},
    "duplicate_rows": {"fn": duplicate_rows, "requires_numeric": False, "requires_string": False},
    "whitespace_noise": {"fn": whitespace_noise, "requires_numeric": False, "requires_string": True},
    "format_inconsistency": {"fn": format_inconsistency, "requires_numeric": False, "requires_string": True},
    "outlier_injection": {"fn": outlier_injection, "requires_numeric": True, "requires_string": False},
    "drop_rows": {"fn": drop_rows, "requires_numeric": False, "requires_string": False},
    "decimal_shift": {"fn": decimal_shift, "requires_numeric": True, "requires_string": False},
    "value_swap": {"fn": value_swap, "requires_numeric": True, "requires_string": False},
    "typo_injection": {"fn": typo_injection, "requires_numeric": False, "requires_string": True},
    "date_format_mix": {"fn": date_format_mix, "requires_numeric": False, "requires_string": True},
    "abbreviation_mix": {"fn": abbreviation_mix, "requires_numeric": False, "requires_string": True},
    "leading_zero_strip": {"fn": leading_zero_strip, "requires_numeric": False, "requires_string": True},
    "header_in_data": {"fn": header_in_data, "requires_numeric": False, "requires_string": False},
    "category_misspell": {"fn": category_misspell, "requires_numeric": False, "requires_string": True},
    "business_rule_violation": {"fn": business_rule_violation, "requires_numeric": True, "requires_string": False},
    "encoding_noise": {"fn": encoding_noise, "requires_numeric": False, "requires_string": True},
    "schema_drift": {"fn": schema_drift, "requires_numeric": False, "requires_string": False},
    "unicode_homoglyph": {"fn": unicode_homoglyph, "requires_numeric": False, "requires_string": True},
    "html_entity_leak": {"fn": html_entity_leak, "requires_numeric": False, "requires_string": True},
    "column_shift": {"fn": column_shift, "requires_numeric": False, "requires_string": False},
    "unit_inconsistency": {"fn": unit_inconsistency, "requires_numeric": True, "requires_string": False},
}
