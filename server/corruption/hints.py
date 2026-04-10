"""Hint generation system with 3 granularity levels for all 22 corruption types.

Levels:
    strategy (easy)  -- Most detailed: column names, counts, code suggestions.
    tactical (medium) -- Column names and counts, no code suggestions.
    categorical (hard) -- Vague: approximate counts, no column names.
"""
from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _approx(n: int) -> int:
    """Round *n* to the nearest 5 for vague hints."""
    if n <= 0:
        return 0
    return max(5, round(n / 5) * 5)


def _col_count_str(cols: dict[str, int]) -> str:
    """Format ``{col: count}`` as ``'Age (5), Fare (3)'``."""
    return ", ".join(f"{col} ({cnt})" for col, cnt in cols.items())


# ---------------------------------------------------------------------------
# Per-corruption template functions
# ---------------------------------------------------------------------------
# Each takes (corruption, cols, total, level, col_stats) and returns str.
# *cols*: {column_name: count_of_errors}
# *col_stats*: optional {column_name: {"mean": ..., "median": ...}}

def _hint_inject_nulls(corruption: str, cols: dict[str, int], total: int,
                       level: str, col_stats: dict | None) -> str:
    if level == "strategy":
        parts = []
        for col, cnt in cols.items():
            desc = f"{col} ({cnt})"
            if col_stats and col in col_stats:
                s = col_stats[col]
                m = s.get("mean")
                md = s.get("median")
                if m is not None and md is not None:
                    desc += f" (mean={m}, median={md})"
            parts.append(desc)
        return (f"Null values in: {', '.join(parts)}. "
                "Try df.fillna() with median/mean.")
    if level == "tactical":
        return f"Null values in: {_col_count_str(cols)}"
    return f"~{_approx(total)} null/missing values detected"


def _hint_type_mangle(corruption: str, cols: dict[str, int], total: int,
                      level: str, col_stats: dict | None) -> str:
    if level == "strategy":
        return (f"Type-mangled values in: {_col_count_str(cols)}. "
                "Try pd.to_numeric(errors='coerce') to recover numeric columns.")
    if level == "tactical":
        return f"Type-mangled values in: {_col_count_str(cols)}"
    return f"~{_approx(total)} values have wrong types"


def _hint_duplicate_rows(corruption: str, cols: dict[str, int], total: int,
                         level: str, col_stats: dict | None) -> str:
    if level == "strategy":
        return (f"{total} duplicate rows detected. "
                "Try df.drop_duplicates() to remove them.")
    if level == "tactical":
        return f"{total} duplicate rows detected"
    return f"~{_approx(total)} duplicate rows detected"


def _hint_whitespace_noise(corruption: str, cols: dict[str, int], total: int,
                           level: str, col_stats: dict | None) -> str:
    if level == "strategy":
        return (f"Whitespace noise in: {_col_count_str(cols)}. "
                "Try df[col].str.strip() on string columns.")
    if level == "tactical":
        return f"Whitespace noise in: {_col_count_str(cols)}"
    return f"~{_approx(total)} values have extra whitespace"


def _hint_format_inconsistency(corruption: str, cols: dict[str, int], total: int,
                               level: str, col_stats: dict | None) -> str:
    if level == "strategy":
        return (f"Format inconsistencies in: {_col_count_str(cols)}. "
                "Standardize formats within each column.")
    if level == "tactical":
        return f"Format inconsistencies in: {_col_count_str(cols)}"
    return f"~{_approx(total)} format inconsistencies detected"


def _hint_outlier_injection(corruption: str, cols: dict[str, int], total: int,
                            level: str, col_stats: dict | None) -> str:
    if level == "strategy":
        return (f"Outliers injected in: {_col_count_str(cols)}. "
                "Detect values beyond +/-3 sigma and cap/clip to column bounds.")
    if level == "tactical":
        return f"Outlier values in: {_col_count_str(cols)}"
    return f"~{_approx(total)} outlier values detected"


def _hint_drop_rows(corruption: str, cols: dict[str, int], total: int,
                    level: str, col_stats: dict | None) -> str:
    if level == "strategy":
        return f"{total} rows appear to be missing. Compare row count against expected length."
    if level == "tactical":
        return f"{total} rows appear to be missing"
    return f"~{_approx(total)} rows may be missing"


def _hint_decimal_shift(corruption: str, cols: dict[str, int], total: int,
                        level: str, col_stats: dict | None) -> str:
    if level == "strategy":
        return (f"Decimal shifts in: {_col_count_str(cols)}. "
                "Values may be off by factors of 10/100/1000.")
    if level == "tactical":
        return f"Decimal shifts in: {_col_count_str(cols)}"
    return f"~{_approx(total)} values have shifted decimals"


def _hint_value_swap(corruption: str, cols: dict[str, int], total: int,
                     level: str, col_stats: dict | None) -> str:
    if level == "strategy":
        return (f"Swapped values in: {_col_count_str(cols)}. "
                "Values may be in wrong rows within the same column.")
    if level == "tactical":
        return f"Swapped values in: {_col_count_str(cols)}"
    return f"~{_approx(total)} values appear to be in wrong rows"


def _hint_typo_injection(corruption: str, cols: dict[str, int], total: int,
                         level: str, col_stats: dict | None) -> str:
    if level == "strategy":
        return (f"Typos in: {_col_count_str(cols)}. "
                "Check for character-level errors in string values.")
    if level == "tactical":
        return f"Typos in: {_col_count_str(cols)}"
    return f"~{_approx(total)} character-level typos detected"


def _hint_date_format_mix(corruption: str, cols: dict[str, int], total: int,
                          level: str, col_stats: dict | None) -> str:
    if level == "strategy":
        return (f"Mixed date formats in: {_col_count_str(cols)}. "
                "Try pd.to_datetime() with infer_datetime_format=True.")
    if level == "tactical":
        return f"Mixed date formats in: {_col_count_str(cols)}"
    return f"~{_approx(total)} date format inconsistencies detected"


def _hint_abbreviation_mix(corruption: str, cols: dict[str, int], total: int,
                           level: str, col_stats: dict | None) -> str:
    if level == "strategy":
        return (f"Abbreviation mix in: {_col_count_str(cols)}. "
                "Standardize to full or abbreviated form consistently.")
    if level == "tactical":
        return f"Abbreviation mix in: {_col_count_str(cols)}"
    return f"~{_approx(total)} inconsistent abbreviations detected"


def _hint_leading_zero_strip(corruption: str, cols: dict[str, int], total: int,
                             level: str, col_stats: dict | None) -> str:
    if level == "strategy":
        return (f"Leading zeros stripped in: {_col_count_str(cols)}. "
                "Try str.zfill() to restore expected zero-padding.")
    if level == "tactical":
        return f"Leading zeros stripped in: {_col_count_str(cols)}"
    return f"~{_approx(total)} values have lost leading zeros"


def _hint_header_in_data(corruption: str, cols: dict[str, int], total: int,
                         level: str, col_stats: dict | None) -> str:
    if level == "strategy":
        return (f"Header rows duplicated in data ({total} occurrences). "
                "Remove rows matching column names.")
    if level == "tactical":
        return f"Header rows duplicated in data ({total} occurrences)"
    return f"~{_approx(total)} rows may be duplicated header lines"


def _hint_category_misspell(corruption: str, cols: dict[str, int], total: int,
                            level: str, col_stats: dict | None) -> str:
    if level == "strategy":
        return (f"Misspelled categories in: {_col_count_str(cols)}. "
                "Use value_counts() to find near-duplicates and standardize.")
    if level == "tactical":
        return f"Misspelled categories in: {_col_count_str(cols)}"
    return f"~{_approx(total)} category labels may be misspelled"


def _hint_business_rule_violation(corruption: str, cols: dict[str, int], total: int,
                                  level: str, col_stats: dict | None) -> str:
    if level == "strategy":
        return (f"Business rule violations in: {_col_count_str(cols)}. "
                "Check for impossible values (negative ages, future dates, etc.).")
    if level == "tactical":
        return f"Business rule violations in: {_col_count_str(cols)}"
    return f"~{_approx(total)} values violate business rules"


def _hint_encoding_noise(corruption: str, cols: dict[str, int], total: int,
                         level: str, col_stats: dict | None) -> str:
    if level == "strategy":
        return (f"Encoding noise (mojibake) in: {_col_count_str(cols)}. "
                "Fix mojibake with str.encode/decode (e.g. latin-1 -> utf-8).")
    if level == "tactical":
        return f"Encoding noise in: {_col_count_str(cols)}"
    return f"~{_approx(total)} values have encoding artifacts"


def _hint_schema_drift(corruption: str, cols: dict[str, int], total: int,
                       level: str, col_stats: dict | None) -> str:
    if level == "strategy":
        return (f"Schema drift in: {_col_count_str(cols)}. "
                "Column names change mid-dataset; normalize headers.")
    if level == "tactical":
        return f"Schema drift in: {_col_count_str(cols)}"
    return f"~{_approx(total)} schema inconsistencies detected"


def _hint_unicode_homoglyph(corruption: str, cols: dict[str, int], total: int,
                            level: str, col_stats: dict | None) -> str:
    if level == "strategy":
        return (f"Unicode homoglyphs in: {_col_count_str(cols)}. "
                "Look-alike characters replacing ASCII; normalize with unicodedata/NFKD.")
    if level == "tactical":
        return f"Unicode homoglyphs in: {_col_count_str(cols)}"
    return f"~{_approx(total)} values contain look-alike characters"


def _hint_html_entity_leak(corruption: str, cols: dict[str, int], total: int,
                           level: str, col_stats: dict | None) -> str:
    if level == "strategy":
        return (f"HTML entity leaks in: {_col_count_str(cols)}. "
                "Replace &amp; &lt; etc. with html.unescape() or str.replace().")
    if level == "tactical":
        return f"HTML entity leaks in: {_col_count_str(cols)}"
    return f"~{_approx(total)} values contain HTML entities"


def _hint_column_shift(corruption: str, cols: dict[str, int], total: int,
                       level: str, col_stats: dict | None) -> str:
    if level == "strategy":
        return (f"Column shifts in: {_col_count_str(cols)}. "
                "Values appear to be circularly shifted across columns.")
    if level == "tactical":
        return f"Column shifts in: {_col_count_str(cols)}"
    return f"~{_approx(total)} values appear shifted across columns"


def _hint_unit_inconsistency(corruption: str, cols: dict[str, int], total: int,
                             level: str, col_stats: dict | None) -> str:
    if level == "strategy":
        return (f"Unit inconsistencies in: {_col_count_str(cols)}. "
                "Standardize to one unit system per column (e.g. all metric or all imperial).")
    if level == "tactical":
        return f"Unit inconsistencies in: {_col_count_str(cols)}"
    return f"~{_approx(total)} values use inconsistent units"


# ---------------------------------------------------------------------------
# Template registry  -- maps corruption name -> template function
# ---------------------------------------------------------------------------

HINT_TEMPLATES: dict[str, Any] = {
    "inject_nulls": _hint_inject_nulls,
    "type_mangle": _hint_type_mangle,
    "duplicate_rows": _hint_duplicate_rows,
    "whitespace_noise": _hint_whitespace_noise,
    "format_inconsistency": _hint_format_inconsistency,
    "outlier_injection": _hint_outlier_injection,
    "drop_rows": _hint_drop_rows,
    "decimal_shift": _hint_decimal_shift,
    "value_swap": _hint_value_swap,
    "typo_injection": _hint_typo_injection,
    "date_format_mix": _hint_date_format_mix,
    "abbreviation_mix": _hint_abbreviation_mix,
    "leading_zero_strip": _hint_leading_zero_strip,
    "header_in_data": _hint_header_in_data,
    "category_misspell": _hint_category_misspell,
    "business_rule_violation": _hint_business_rule_violation,
    "encoding_noise": _hint_encoding_noise,
    "schema_drift": _hint_schema_drift,
    "unicode_homoglyph": _hint_unicode_homoglyph,
    "html_entity_leak": _hint_html_entity_leak,
    "column_shift": _hint_column_shift,
    "unit_inconsistency": _hint_unit_inconsistency,
}


# ---------------------------------------------------------------------------
# Main public API
# ---------------------------------------------------------------------------

def generate_hints(
    error_map: dict[str, Any],
    hint_level: str,
    col_stats: dict[str, dict] | None = None,
) -> str:
    """Generate human-readable hints from an error map at the given granularity.

    Args:
        error_map: Dict with keys ``"cell_errors"``, ``"spurious_rows"``,
            ``"missing_rows"``.  ``cell_errors`` maps ``"row,col"`` to
            ``{"severity": N, "clean_value": V, "corruption": "type"}``.
        hint_level: One of ``"strategy"`` (easy), ``"tactical"`` (medium),
            ``"categorical"`` (hard).
        col_stats: Optional per-column statistics (mean, median, etc.)
            used only at the strategy level.

    Returns:
        Multi-line string with one hint per corruption type, plus sections
        for spurious and missing rows when present.
    """
    cell_errors: dict[str, dict] = error_map.get("cell_errors", {})
    spurious_rows: dict[str, dict] = error_map.get("spurious_rows", {})
    missing_rows: dict[str, dict] = error_map.get("missing_rows", {})

    # ---- Group cell errors by corruption type and column ----
    # {corruption_type: {column_name: count}}
    grouped: dict[str, dict[str, int]] = {}
    for key, err in cell_errors.items():
        corruption = err.get("corruption", "unknown")
        # key format is "row,col"
        parts = key.split(",", 1)
        col = parts[1] if len(parts) == 2 else "unknown"
        grouped.setdefault(corruption, {})
        grouped[corruption][col] = grouped[corruption].get(col, 0) + 1

    lines: list[str] = []

    # ---- Corruption-type hints ----
    for corruption, cols in sorted(grouped.items()):
        total = sum(cols.values())
        template_fn = HINT_TEMPLATES.get(corruption)
        if template_fn is not None:
            lines.append(template_fn(corruption, cols, total, hint_level, col_stats))
        else:
            # Fallback for unknown corruption types
            if hint_level == "strategy":
                lines.append(f"Unknown corruption '{corruption}' in: {_col_count_str(cols)}")
            elif hint_level == "tactical":
                lines.append(f"Unknown corruption in: {_col_count_str(cols)}")
            else:
                lines.append(f"~{_approx(total)} unknown data issues detected")

    # ---- Spurious rows ----
    if spurious_rows:
        n = len(spurious_rows)
        if hint_level == "strategy":
            row_ids = ", ".join(sorted(spurious_rows.keys(), key=lambda k: int(k) if k.isdigit() else k))
            lines.append(f"{n} spurious (extra) rows at indices: {row_ids}. "
                         "Remove them with df.drop().")
        elif hint_level == "tactical":
            lines.append(f"{n} spurious (extra) rows detected")
        else:
            lines.append(f"~{_approx(n)} extra rows that should not be present")

    # ---- Missing rows ----
    if missing_rows:
        n = len(missing_rows)
        if hint_level == "strategy":
            lines.append(f"{n} rows are missing from the dataset. "
                         "Restore them from a backup or reconstruct from context.")
        elif hint_level == "tactical":
            lines.append(f"{n} rows are missing from the dataset")
        else:
            lines.append(f"~{_approx(n)} rows appear to be missing")

    return "\n".join(lines)


def generate_format_hints(
    format_metadata: list[dict[str, Any]],
    hint_level: str,
) -> str:
    """Generate hints for format-level corruptions.

    Args:
        format_metadata: List of dicts, each describing one format corruption.
            Expected keys include ``"type"`` and optionally ``"details"``.
        hint_level: One of ``"strategy"``, ``"tactical"``, ``"categorical"``.

    Returns:
        Multi-line string of format hints.
    """
    if not format_metadata:
        return ""

    if hint_level == "categorical":
        return "Some file-format issues detected"

    lines: list[str] = []
    for meta in format_metadata:
        fmt_type = meta.get("type", "unknown")
        details = meta.get("details", "")
        if hint_level == "strategy":
            detail_str = f": {details}" if details else ""
            lines.append(f"Format issue ({fmt_type}){detail_str}")
        else:
            # tactical
            lines.append(f"Format issue: {fmt_type}")

    return "\n".join(lines)
