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

import numpy as np
import pandas as pd


# ── Constants ────────────────────────────────────────────────────────────────

WRONG_VALUE_MULTIPLIER = 1.5  # penalty for changing a cell to the wrong value
COLLATERAL_DAMAGE_WEIGHT = 0.5  # severity per incorrectly changed clean cell
NEAR_MISS_THRESHOLD = 0.05  # relative tolerance for partial credit on numeric


# ── Core Grading ─────────────────────────────────────────────────────────────


def _values_equal(a: Any, b: Any) -> bool:
    """Compare two cell values with tolerance for numeric types and NaN."""
    try:
        a_na, b_na = pd.isna(a), pd.isna(b)
        if a_na and b_na:
            # Both missing — but reject cross-type NaN (NaT vs NaN)
            a_is_nat = isinstance(a, pd.Timestamp) or type(a).__name__ == "NaTType"
            b_is_nat = isinstance(b, pd.Timestamp) or type(b).__name__ == "NaTType"
            if a_is_nat != b_is_nat:
                return False
            return True
        if a_na or b_na:
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


def _numeric_distance(a: Any, b: Any) -> float | None:
    """Return relative distance between two numeric values, or None if non-numeric."""
    try:
        fa, fb = float(a), float(b)
        if math.isinf(fa) or math.isinf(fb) or math.isnan(fa) or math.isnan(fb):
            return None
        denom = max(abs(fb), 1e-9)
        return abs(fa - fb) / denom
    except (TypeError, ValueError):
        return None


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

    # Numeric column: accept if within [min - margin, max + margin]
    # Use stddev-based margin instead of fixed 10% of range
    try:
        fval = float(result_val)
        numeric = pd.to_numeric(series, errors="coerce").dropna()
        if len(numeric) > 0:
            col_min, col_max = numeric.min(), numeric.max()
            col_std = numeric.std()
            margin = max(col_std * 0.5, 1e-6) if col_std > 0 else max(abs(col_max) * 0.1, 1.0)
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


def _parse_error_key(key: str) -> tuple[str, str]:
    """Parse 'row,col' error key. Handles column names containing commas."""
    # The key format is "row_index,column_name" — row_index is always an integer,
    # so split on the first comma only
    idx = key.index(",")
    return key[:idx], key[idx + 1 :]


def _detect_collateral_damage(
    clean_df: pd.DataFrame,
    result_df: pd.DataFrame,
    error_map: dict[str, Any],
) -> float:
    """Detect cells that were correct in dirty data but got corrupted by the agent.

    Returns total collateral severity to add to remaining_severity.
    """
    cell_errors = error_map.get("cell_errors", {})
    # Build set of (row, col) that are known errors — skip these
    error_cells: set[tuple[int, str]] = set()
    for key in cell_errors:
        try:
            row_str, col = _parse_error_key(key)
            error_cells.add((int(row_str), col))
        except (ValueError, IndexError):
            pass

    collateral = 0.0
    # Only check columns and rows present in both clean and result
    shared_cols = set(clean_df.columns) & set(result_df.columns)
    shared_idx = set(clean_df.index) & set(result_df.index)

    for col in shared_cols:
        clean_col = clean_df[col]
        result_col = result_df[col]
        for idx in shared_idx:
            if (idx, col) in error_cells:
                continue
            clean_val = clean_col.at[idx]
            try:
                result_val = result_col.at[idx]
            except KeyError:
                continue
            if not _values_equal(clean_val, result_val):
                collateral += COLLATERAL_DAMAGE_WEIGHT

    return collateral


def _check_spurious_row(
    clean_df: pd.DataFrame,
    result_df: pd.DataFrame,
    row_str: str,
) -> bool:
    """Check if a spurious (duplicate) row is still present in the result.

    Spurious rows are extra rows appended beyond the clean data length.
    Fixed = result doesn't have that row index, OR result is same length as clean.
    """
    try:
        row_idx = int(row_str)
    except ValueError:
        return False  # can't parse → assume still present

    # If the row index doesn't exist in result, it's been removed
    if row_idx not in result_df.index:
        return True  # fixed

    # If result has been trimmed to clean length or shorter, spurious rows are gone
    if len(result_df) <= len(clean_df):
        return True  # fixed

    return False  # still present


def _check_missing_row(
    clean_df: pd.DataFrame,
    result_df: pd.DataFrame,
    row_str: str,
) -> str:
    """Check if a missing row has been restored with correct content.

    Returns: 'fixed', 'wrong_value', or 'unfixed'.
    """
    try:
        row_idx = int(row_str)
    except ValueError:
        return "unfixed"

    if row_idx not in result_df.index:
        return "unfixed"

    # Row exists — verify content matches clean data
    if row_idx not in clean_df.index:
        return "fixed"  # can't verify, benefit of doubt

    clean_row = clean_df.loc[row_idx]
    result_row = result_df.loc[row_idx]
    matches = 0
    total = 0
    for col in clean_df.columns:
        if col not in result_df.columns:
            continue
        total += 1
        if _values_equal(clean_row.get(col), result_row.get(col)):
            matches += 1

    if total == 0:
        return "fixed"
    match_ratio = matches / total
    if match_ratio >= 0.8:
        return "fixed"
    elif match_ratio >= 0.3:
        return "wrong_value"  # partially restored
    return "unfixed"


def grade(
    clean_df: pd.DataFrame,
    result_df: pd.DataFrame,
    error_map: dict[str, Any],
    transform_steps: int,
    min_transform_steps: int,
    max_transform_steps: int,
    explore_steps: int = 0,
    explore_timeouts: int = 0,
    explore_cost_per_step: float = 0.01,
    explore_timeout_cost: float = 0.03,
) -> tuple[dict[str, str], float]:
    """Grade the agent's result against the clean reference using the error map.

    Args:
        clean_df: Ground truth DataFrame.
        result_df: Agent's cleaned DataFrame.
        error_map: Dict with keys "cell_errors", "spurious_rows", "missing_rows".
        transform_steps: Number of transform steps the agent used.
        min_transform_steps: Minimum expected steps (efficiency baseline).
        max_transform_steps: Maximum allowed steps.
        explore_steps: Total number of explore actions taken.
        explore_timeouts: Number of explore actions that timed out or failed.
        explore_cost_per_step: Penalty per successful explore action.
        explore_timeout_cost: Penalty per timed-out/failed explore action.

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
        row_str, col = _parse_error_key(key)
        severity = float(info["severity"])
        clean_val = info["clean_value"]
        total_severity += severity

        try:
            row_idx = int(row_str)
        except ValueError:
            error_status[key] = "unfixed"
            remaining_severity += severity
            continue

        if row_str not in result_index:
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
                    if _is_reasonable_fill(clean_df, col, result_val):
                        error_status[key] = "fixed"
                    else:
                        error_status[key] = "wrong_value"
                        remaining_severity += severity * WRONG_VALUE_MULTIPLIER
                elif accepted_fill == "exact":
                    error_status[key] = "wrong_value"
                    remaining_severity += severity * WRONG_VALUE_MULTIPLIER
                elif accepted_fill in ("mean", "median", "mode"):
                    if _check_stat_fill(clean_df, col, result_val, accepted_fill):
                        error_status[key] = "fixed"
                    else:
                        error_status[key] = "wrong_value"
                        remaining_severity += severity * WRONG_VALUE_MULTIPLIER
                else:
                    # Unknown accepted_fill or None — default to "any"
                    error_status[key] = "fixed"
            elif corruption_type in ("whitespace_noise", "format_inconsistency"):
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
                # Other corruptions (type_mangle, outlier, etc.)
                # Graduated penalty: near-miss gets partial credit
                dist = _numeric_distance(result_val, clean_val)
                if dist is not None and dist <= NEAR_MISS_THRESHOLD:
                    # Close enough — partial penalty instead of full wrong_value
                    # Scale: 0 distance = 0 penalty, threshold = full wrong penalty
                    fraction = dist / NEAR_MISS_THRESHOLD
                    error_status[key] = "wrong_value"
                    remaining_severity += severity * fraction * WRONG_VALUE_MULTIPLIER
                else:
                    error_status[key] = "wrong_value"
                    remaining_severity += severity * WRONG_VALUE_MULTIPLIER

    # ── Spurious rows (duplicates) ───────────────────────────────────────────
    for row_str, info in spurious_rows.items():
        severity = float(info["severity"])
        total_severity += severity
        key = f"spurious_{row_str}"

        if _check_spurious_row(clean_df, result_df, row_str):
            error_status[key] = "fixed"
        else:
            error_status[key] = "unfixed"
            remaining_severity += severity

    # ── Missing rows ─────────────────────────────────────────────────────────
    for row_str, info in missing_rows.items():
        severity = float(info["severity"])
        total_severity += severity
        key = f"missing_{row_str}"

        status = _check_missing_row(clean_df, result_df, row_str)
        error_status[key] = status
        if status == "unfixed":
            remaining_severity += severity
        elif status == "wrong_value":
            remaining_severity += severity * 0.5  # partial credit for attempt

    # ── Collateral damage ────────────────────────────────────────────────────
    collateral_severity = _detect_collateral_damage(clean_df, result_df, error_map)
    total_severity += collateral_severity
    remaining_severity += collateral_severity

    # ── Compute reward ───────────────────────────────────────────────────────
    if total_severity == 0.0:
        return error_status, 1.0

    # Clamp remaining_severity (wrong_value penalty can push it above total)
    constraint_score = max(0.0, 1.0 - remaining_severity / total_severity)

    # Transform penalty
    transform_excess = max(0, transform_steps - min_transform_steps)
    transform_penalty = transform_excess / (max_transform_steps * 2)

    # Explore penalty: normal explores cost a little, timed-out ones cost more
    normal_explores = explore_steps - explore_timeouts
    explore_penalty = (normal_explores * explore_cost_per_step) + (explore_timeouts * explore_timeout_cost)

    efficiency_factor = max(0.5, 1.0 - transform_penalty - explore_penalty)

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
