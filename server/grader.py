"""
Generic diff-based grader for data cleaning tasks.

The grader receives clean data, result data, and a pre-built error map
(produced by the corruption engine). It computes a severity-weighted score
based on how many errors were fixed, with an extra penalty for cells changed
to the wrong value, and an efficiency factor based on transform steps used.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd


# ── Constants ────────────────────────────────────────────────────────────────

WRONG_VALUE_MULTIPLIER = 1.5  # penalty for changing a cell to the wrong value


# ── Core Grading ─────────────────────────────────────────────────────────────


def _values_equal(a: Any, b: Any) -> bool:
    """Compare two cell values with tolerance for numeric types and NaN."""
    try:
        if pd.isna(a) and pd.isna(b):
            return True
        if pd.isna(a) or pd.isna(b):
            return False
    except (TypeError, ValueError):
        pass
    # Numeric tolerance — handles float/int/string-of-number mismatches
    try:
        fa, fb = float(a), float(b)
        return abs(fa - fb) < 1e-6
    except (TypeError, ValueError):
        pass
    # Exact string comparison — do NOT strip, whitespace corruption is intentional
    return str(a) == str(b)


def _is_reasonable_fill(clean_df: pd.DataFrame, col: str, result_val: Any) -> bool:
    """Check if a fill value is a reasonable imputation (not random garbage).

    Accepts: any value within the column's observed range for numeric columns,
    or any value that exists in the column for categorical columns.
    This covers mean, median, mode, ffill, bfill, interpolate, and similar.
    """
    if col not in clean_df.columns:
        return True  # can't validate, give benefit of doubt
    series = clean_df[col].dropna()
    if len(series) == 0:
        return True

    # Numeric column: accept if within [min - 10% range, max + 10% range]
    try:
        fval = float(result_val)
        numeric = pd.to_numeric(series, errors="coerce").dropna()
        if len(numeric) > 0:
            col_min, col_max = numeric.min(), numeric.max()
            col_range = col_max - col_min if col_max != col_min else abs(col_max) * 0.1 or 1.0
            margin = col_range * 0.1
            return (col_min - margin) <= fval <= (col_max + margin)
    except (TypeError, ValueError):
        pass

    # Categorical column: accept if value exists in column's unique values
    unique_vals = set(series.astype(str).str.strip().str.lower())
    try:
        return str(result_val).strip().lower() in unique_vals
    except Exception:
        pass

    return False  # can't interpret → reject


def _check_stat_fill(clean_df: pd.DataFrame, col: str, result_val: Any, stat: str) -> bool:
    """Check if result_val matches the expected column statistic from clean data."""
    if col not in clean_df.columns:
        return False
    try:
        if stat == "mean":
            expected = clean_df[col].mean()
        elif stat == "median":
            expected = clean_df[col].median()
        elif stat == "mode":
            mode_vals = clean_df[col].mode()
            if len(mode_vals) == 0:
                return False
            # Accept any of the modes
            return any(_values_equal(result_val, m) for m in mode_vals)
        else:
            return False
        return _values_equal(result_val, expected)
    except Exception:
        return False


def grade(
    clean_df: pd.DataFrame,
    result_df: pd.DataFrame,
    error_map: dict[str, Any],
    transform_steps: int,
    min_transform_steps: int,
    max_transform_steps: int,
) -> tuple[dict[str, str], float]:
    """Grade the agent's result against the clean reference using the error map.

    Args:
        clean_df: Ground truth DataFrame.
        result_df: Agent's cleaned DataFrame.
        error_map: Dict with keys "cell_errors", "spurious_rows", "missing_rows".
        transform_steps: Number of transform steps the agent used.
        min_transform_steps: Minimum expected steps (efficiency baseline).
        max_transform_steps: Maximum allowed steps.

    Returns:
        (error_status, reward) where error_status maps each error key to one of
        "fixed", "wrong_value", or "unfixed", and reward is in [0.0, 1.0].
    """
    cell_errors: dict[str, dict] = error_map.get("cell_errors", {})
    spurious_rows: dict[str, dict] = error_map.get("spurious_rows", {})
    missing_rows: dict[str, dict] = error_map.get("missing_rows", {})

    total_severity = 0.0
    remaining_severity = 0.0
    error_status: dict[str, str] = {}

    # ── Cell errors ──────────────────────────────────────────────────────────
    result_index = set(result_df.index.astype(str))

    for key, info in cell_errors.items():
        row_str, col = key.rsplit(",", 1)
        severity = float(info["severity"])
        clean_val = info["clean_value"]
        total_severity += severity

        try:
            row_idx = int(row_str)
        except ValueError:
            # Can't match row — treat as unfixed
            error_status[key] = "unfixed"
            remaining_severity += severity
            continue

        if row_str not in result_index:
            # Row missing from result entirely
            error_status[key] = "unfixed"
            remaining_severity += severity
            continue

        if col not in result_df.columns:
            error_status[key] = "unfixed"
            remaining_severity += severity
            continue

        try:
            result_val = result_df.at[row_idx, col]
        except KeyError:
            error_status[key] = "unfixed"
            remaining_severity += severity
            continue

        if _values_equal(result_val, clean_val):
            error_status[key] = "fixed"
        else:
            corruption_type = info.get("corruption", "")
            accepted_fill = info.get("accepted_fill")

            try:
                result_is_nan = pd.isna(result_val)
            except (TypeError, ValueError):
                result_is_nan = False

            if corruption_type == "null_injected":
                if result_is_nan:
                    # Still null — unfixed
                    error_status[key] = "unfixed"
                    remaining_severity += severity
                elif accepted_fill == "any":
                    # Any reasonable imputation is acceptable (mean, median, mode,
                    # ffill, bfill, interpolate). Reject obvious garbage.
                    if _is_reasonable_fill(clean_df, col, result_val):
                        error_status[key] = "fixed"
                    else:
                        error_status[key] = "wrong_value"
                        remaining_severity += severity * WRONG_VALUE_MULTIPLIER
                elif accepted_fill == "exact":
                    # Must match clean value exactly — already failed above
                    error_status[key] = "wrong_value"
                    remaining_severity += severity * WRONG_VALUE_MULTIPLIER
                elif accepted_fill in ("mean", "median", "mode"):
                    # Validate against the column statistic
                    if _check_stat_fill(clean_df, col, result_val, accepted_fill):
                        error_status[key] = "fixed"
                    else:
                        error_status[key] = "wrong_value"
                        remaining_severity += severity * WRONG_VALUE_MULTIPLIER
                else:
                    # Unknown accepted_fill or None — default to "any"
                    error_status[key] = "fixed"
            elif corruption_type in ("whitespace_noise", "format_inconsistency"):
                # Whitespace/format: accept if stripped version matches clean value
                try:
                    result_stripped = str(result_val).strip()
                    clean_stripped = str(clean_val).strip()
                    if result_stripped.lower() == clean_stripped.lower():
                        error_status[key] = "fixed"
                    else:
                        error_status[key] = "wrong_value"
                        remaining_severity += severity * WRONG_VALUE_MULTIPLIER
                except Exception:
                    error_status[key] = "wrong_value"
                    remaining_severity += severity * WRONG_VALUE_MULTIPLIER
            else:
                # Other corruptions (type_mangle, outlier)
                # Changed to wrong value → penalized
                error_status[key] = "wrong_value"
                remaining_severity += severity * WRONG_VALUE_MULTIPLIER

    # ── Spurious rows (duplicates) ───────────────────────────────────────────
    clean_len = len(clean_df)
    result_len = len(result_df)

    for row_str, info in spurious_rows.items():
        severity = float(info["severity"])
        total_severity += severity
        key = f"spurious_{row_str}"

        # Heuristic: if result has more rows than clean, spurious rows remain
        if result_len > clean_len:
            error_status[key] = "unfixed"
            remaining_severity += severity
        else:
            error_status[key] = "fixed"

    # ── Missing rows ─────────────────────────────────────────────────────────
    for row_str, info in missing_rows.items():
        severity = float(info["severity"])
        total_severity += severity
        key = f"missing_{row_str}"

        try:
            row_idx = int(row_str)
            if row_idx in result_df.index:
                error_status[key] = "fixed"
            else:
                error_status[key] = "unfixed"
                remaining_severity += severity
        except (ValueError, KeyError):
            error_status[key] = "unfixed"
            remaining_severity += severity

    # ── Compute reward ───────────────────────────────────────────────────────
    if total_severity == 0.0:
        return error_status, 1.0

    # Clamp remaining_severity (wrong_value penalty can push it above total)
    constraint_score = max(0.0, 1.0 - remaining_severity / total_severity)

    excess_steps = max(0, transform_steps - min_transform_steps)
    efficiency_factor = max(0.5, 1.0 - excess_steps / (max_transform_steps * 2))

    reward = round(min(1.0, constraint_score * efficiency_factor), 4)
    return error_status, reward


# ── Summary Helper ───────────────────────────────────────────────────────────


def summarize_errors(error_status: dict[str, str], error_map: dict[str, Any]) -> dict[str, Any]:
    """Summarize error status counts for logging and observation building."""
    total = len(error_status)
    fixed = sum(1 for s in error_status.values() if s == "fixed")
    wrong = sum(1 for s in error_status.values() if s == "wrong_value")
    unfixed = sum(1 for s in error_status.values() if s == "unfixed")

    cell_errors = error_map.get("cell_errors", {})
    spurious = error_map.get("spurious_rows", {})
    missing = error_map.get("missing_rows", {})

    return {
        "total_errors": total,
        "fixed": fixed,
        "wrong_value": wrong,
        "unfixed": unfixed,
        "cell_errors_total": len(cell_errors),
        "spurious_rows_total": len(spurious),
        "missing_rows_total": len(missing),
    }
