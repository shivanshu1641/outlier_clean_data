"""
Generic diff-based grader for data cleaning tasks.

The grader receives clean data, result data, and a pre-built error map
(produced by the corruption engine). It computes a multi-level score across
schema, row, cell, and distribution dimensions, with an efficiency factor
based on transform steps and exploration overhead.
"""

from __future__ import annotations

import math
from collections import defaultdict
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
    row_mapping: dict[int, int] | None = None,
) -> float:
    """Detect cells that were correct in dirty data but got corrupted by the agent.

    Returns total collateral severity to add to remaining_severity.

    Uses row_mapping to align result rows to clean rows so that reordered
    DataFrames are compared correctly.
    """
    cell_errors = error_map.get("cell_errors", {})

    # Build set of known error cells for fast lookup
    error_cells: set[tuple[int, str]] = set()
    for key in cell_errors:
        try:
            row_str, col = _parse_error_key(key)
            error_cells.add((int(row_str), col))
        except (ValueError, IndexError):
            pass

    shared_cols = set(clean_df.columns) & set(result_df.columns)

    collateral = 0.0
    for clean_idx in clean_df.index:
        # Use row_mapping to find the corresponding result row
        if row_mapping is None:
            result_idx = clean_idx
        else:
            result_idx = row_mapping.get(clean_idx, clean_idx)
        if row_mapping and clean_idx not in row_mapping:
            continue
        if result_idx not in result_df.index:
            continue
        for col in shared_cols:
            if (clean_idx, col) in error_cells:
                continue
            try:
                clean_val = clean_df.at[clean_idx, col]
                result_val = result_df.at[result_idx, col]
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
    Fixed = result doesn't have that row index.
    """
    try:
        row_idx = int(row_str)
    except ValueError:
        return False  # can't parse → assume still present

    if row_idx < 0:
        return False  # invalid row id → assume still present

    # If the row index no longer exists in result, it's been removed
    if row_idx not in result_df.index:
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


# ── Schema Scoring ──────────────────────────────────────────────────────────


def _dtypes_compatible(a, b) -> bool:
    """Check if two dtypes are compatible."""
    a_str, b_str = str(a), str(b)
    numeric = {"int64", "int32", "float64", "float32", "Int64", "Float64"}
    string = {"object", "string", "str"}
    if a_str in numeric and b_str in numeric:
        return True
    if a_str in string and b_str in string:
        return True
    return a_str == b_str


def schema_score(clean_df: pd.DataFrame, result_df: pd.DataFrame) -> float:
    """Score structural correctness of the result DataFrame.

    Checks column name matching (case-insensitive) and type compatibility.
    Column reorder is fine (no penalty). Extra columns don't penalize.

    Returns a score in [0.0, 1.0] (weight: 0.15 in final grade).
    """
    clean_cols = {c.lower(): c for c in clean_df.columns}
    result_cols = {c.lower(): c for c in result_df.columns}

    total_clean = len(clean_cols)
    if total_clean == 0:
        return 1.0

    matched = set(clean_cols) & set(result_cols)
    col_ratio = len(matched) / total_clean

    # Type compatibility for matched columns
    if len(matched) == 0:
        type_compat = 0.0
    else:
        compat_count = 0
        for lower_col in matched:
            clean_dtype = clean_df[clean_cols[lower_col]].dtype
            result_dtype = result_df[result_cols[lower_col]].dtype
            if _dtypes_compatible(clean_dtype, result_dtype):
                compat_count += 1
        type_compat = compat_count / len(matched)

    return col_ratio * 0.7 + type_compat * 0.3


# ── Row Matching ────────────────────────────────────────────────────────────


def _row_hash(row) -> str:
    """Compute a deterministic hash for a row."""
    parts = []
    for val in row:
        if pd.isna(val):
            parts.append("__NA__")
        else:
            parts.append(str(val).strip().lower())
    return "|".join(parts)


def match_rows_by_content(
    clean_df: pd.DataFrame, result_df: pd.DataFrame
) -> dict[int, int]:
    """Content-based row matching. Returns {clean_idx: result_idx}.

    Uses hash-based O(n) matching with collision handling.
    """
    clean_cols = {c.lower(): c for c in clean_df.columns}
    result_cols = {c.lower(): c for c in result_df.columns}
    shared = sorted(set(clean_cols) & set(result_cols))

    if not shared:
        return {}

    # Build hash index for result
    result_hashes: dict[str, list[int]] = defaultdict(list)
    for idx in result_df.index:
        vals = tuple(result_df.at[idx, result_cols[c]] for c in shared)
        h = _row_hash(vals)
        result_hashes[h].append(idx)

    mapping: dict[int, int] = {}
    used_result: set[int] = set()
    for cidx in clean_df.index:
        vals = tuple(clean_df.at[cidx, clean_cols[c]] for c in shared)
        h = _row_hash(vals)
        candidates = result_hashes.get(h, [])
        for ridx in candidates:
            if ridx not in used_result:
                mapping[cidx] = ridx
                used_result.add(ridx)
                break
    return mapping


# ── Row Scoring ─────────────────────────────────────────────────────────────


def row_score(
    clean_df: pd.DataFrame,
    result_df: pd.DataFrame,
    error_map: dict[str, Any],
) -> float:
    """Score row completeness.

    Considers row count ratio, spurious row penalty, and missing row penalty.
    Returns a score in [0.0, 1.0] (weight: 0.20 in final grade).
    """
    expected = len(clean_df)
    actual = len(result_df)

    # Base: row count ratio (penalizes both too many and too few)
    if expected == 0:
        return 1.0
    count_ratio = min(actual, expected) / max(actual, expected)

    # Spurious row penalty
    spurious = error_map.get("spurious_rows", {})
    spurious_still = sum(
        1 for r in spurious if not _check_spurious_row(clean_df, result_df, r)
    )
    spurious_penalty = (
        spurious_still / max(len(spurious), 1) * 0.5 if spurious else 0
    )

    # Missing row penalty
    missing = error_map.get("missing_rows", {})
    missing_still = sum(
        1 for r in missing if _check_missing_row(clean_df, result_df, r) == "unfixed"
    )
    missing_penalty = (
        missing_still / max(len(missing), 1) * 0.5 if missing else 0
    )

    return max(0.0, count_ratio - spurious_penalty - missing_penalty)


# ── Cell Scoring ────────────────────────────────────────────────────────────


def _cell_score_full(
    clean_df: pd.DataFrame,
    result_df: pd.DataFrame,
    error_map: dict[str, Any],
    row_mapping: dict[int, int] | None = None,
) -> tuple[float, dict[str, str]]:
    """Internal: compute cell score and error_status dict together."""
    cell_errors: dict[str, dict] = error_map.get("cell_errors", {})
    spurious_rows: dict[str, dict] = error_map.get("spurious_rows", {})
    missing_rows: dict[str, dict] = error_map.get("missing_rows", {})

    total_severity = 0.0
    remaining_severity = 0.0
    error_status: dict[str, str] = {}

    # ── Cell errors ──────────────────────────────────────────────────────
    result_index = set(result_df.index.astype(str))

    for key, info in cell_errors.items():
        try:
            row_str, col = _parse_error_key(key)
        except ValueError:
            continue
        severity = float(info["severity"])
        clean_val = info["clean_value"]
        total_severity += severity

        try:
            row_idx = int(row_str)
        except ValueError:
            error_status[key] = "unfixed"
            remaining_severity += severity
            continue

        # Use row_mapping to find the actual result row index
        mapped_idx = row_mapping.get(row_idx, row_idx) if row_mapping else row_idx

        if str(mapped_idx) not in result_index:
            error_status[key] = "unfixed"
            remaining_severity += severity
            continue

        if col not in result_df.columns:
            error_status[key] = "unfixed"
            remaining_severity += severity
            continue

        try:
            result_val = result_df.at[mapped_idx, col]
        except KeyError:
            error_status[key] = "unfixed"
            remaining_severity += severity
            continue

        corruption_type = info.get("corruption", "")
        accepted_fill = info.get("accepted_fill")

        if corruption_type in ("whitespace_noise", "format_inconsistency"):
            # Whitespace corruption: require exact string match — float() strips
            # whitespace so _values_equal would give false positives
            if str(result_val) == str(clean_val):
                error_status[key] = "fixed"
            else:
                try:
                    result_stripped = str(result_val).strip()
                    clean_stripped = str(clean_val).strip()
                    if result_stripped.lower() == clean_stripped.lower():
                        # Content matches but formatting doesn't → still corrupted
                        error_status[key] = "unfixed"
                        remaining_severity += severity
                    else:
                        error_status[key] = "wrong_value"
                        remaining_severity += severity * WRONG_VALUE_MULTIPLIER
                except Exception:
                    error_status[key] = "wrong_value"
                    remaining_severity += severity * WRONG_VALUE_MULTIPLIER
        elif _values_equal(result_val, clean_val):
            # Cross-check with dirty_value — float() strips leading zeros,
            # coerces type_mangle, etc., giving false "fixed" results.
            # Use repr() to preserve type info (str("42") == str(42) but repr differs)
            dirty_val = info.get("dirty_value")
            if (dirty_val is not None
                    and repr(result_val) == repr(dirty_val)
                    and repr(clean_val) != repr(dirty_val)):
                error_status[key] = "unfixed"
                remaining_severity += severity
            else:
                error_status[key] = "fixed"
        else:
            dirty_val = info.get("dirty_value")

            try:
                result_is_nan = pd.isna(result_val)
            except (TypeError, ValueError):
                result_is_nan = False
            try:
                clean_is_nan = pd.isna(clean_val)
            except (TypeError, ValueError):
                clean_is_nan = False

            if corruption_type in ("null_injected", "inject_nulls"):
                if result_is_nan:
                    # Still null -- unfixed
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
                    # Unknown accepted_fill or None -- default to "any"
                    error_status[key] = "fixed"
            elif (
                corruption_type == "type_mangle"
                and result_is_nan
                and not clean_is_nan
                and dirty_val is None
            ):
                # Some dirty string sentinels like "NA"/"N/A"/"null" are parsed by pandas
                # as NaN before the agent sees them. If the agent leaves that parsed NaN
                # untouched, it is still the original dirty state, not a new wrong value.
                error_status[key] = "unfixed"
                remaining_severity += severity
            elif dirty_val is not None and _values_equal(result_val, dirty_val):
                # Result still matches dirty value — corruption unfixed
                error_status[key] = "unfixed"
                remaining_severity += severity
            else:
                # Agent changed value to something other than clean or dirty
                dist = _numeric_distance(result_val, clean_val)
                if dist is not None and dist <= NEAR_MISS_THRESHOLD:
                    fraction = dist / NEAR_MISS_THRESHOLD
                    error_status[key] = "wrong_value"
                    remaining_severity += severity * fraction * WRONG_VALUE_MULTIPLIER
                else:
                    error_status[key] = "wrong_value"
                    remaining_severity += severity * WRONG_VALUE_MULTIPLIER

    # ── Spurious rows (duplicates) ───────────────────────────────────────
    for row_str, info in spurious_rows.items():
        severity = float(info["severity"])
        total_severity += severity
        skey = f"spurious_{row_str}"

        if _check_spurious_row(clean_df, result_df, row_str):
            error_status[skey] = "fixed"
        else:
            error_status[skey] = "unfixed"
            remaining_severity += severity

    # ── Missing rows ─────────────────────────────────────────────────────
    for row_str, info in missing_rows.items():
        severity = float(info["severity"])
        total_severity += severity
        mkey = f"missing_{row_str}"

        status = _check_missing_row(clean_df, result_df, row_str)
        error_status[mkey] = status
        if status == "unfixed":
            remaining_severity += severity
        elif status == "wrong_value":
            remaining_severity += severity * 0.5  # partial credit for attempt

    # ── Collateral damage ────────────────────────────────────────────────
    collateral_severity = _detect_collateral_damage(clean_df, result_df, error_map, row_mapping)
    total_severity += collateral_severity
    remaining_severity += collateral_severity

    # ── Score ────────────────────────────────────────────────────────────
    if total_severity == 0.0:
        return 1.0, error_status

    score = max(0.0, 1.0 - remaining_severity / total_severity)
    return score, error_status


def cell_score(
    clean_df: pd.DataFrame,
    result_df: pd.DataFrame,
    error_map: dict[str, Any],
    row_mapping: dict[int, int] | None = None,
) -> float:
    """Score cell-level correctness using the error map.

    Uses row_mapping (from match_rows_by_content) when looking up result
    values, falling back to direct index access.

    Returns score in [0.0, 1.0] (weight: 0.55 in final grade).
    """
    score, _ = _cell_score_full(clean_df, result_df, error_map, row_mapping)
    return score


# ── Distribution Scoring ────────────────────────────────────────────────────


def distribution_score(
    clean_df: pd.DataFrame,
    result_df: pd.DataFrame,
    imputed_cols: set[str] | list[str] | None = None,
) -> float:
    """Score imputation quality by comparing distributions.

    Returns a score in [0.0, 1.0] (weight: 0.10 in final grade).
    """
    if not imputed_cols:
        return 1.0
    distances: list[float] = []
    for col in imputed_cols:
        if col not in clean_df.columns or col not in result_df.columns:
            continue
        try:
            clean_num = pd.to_numeric(clean_df[col], errors="coerce").dropna()
            result_num = pd.to_numeric(result_df[col], errors="coerce").dropna()
            if len(clean_num) < 2 or len(result_num) < 2:
                continue
            col_range = clean_num.max() - clean_num.min()
            if col_range < 1e-9:
                continue
            mean_dist = abs(clean_num.mean() - result_num.mean()) / col_range
            std_dist = abs(clean_num.std() - result_num.std()) / col_range
            median_dist = abs(clean_num.median() - result_num.median()) / col_range
            distances.append(min(1.0, (mean_dist + std_dist + median_dist) / 3))
        except Exception:
            continue
    if not distances:
        return 1.0
    return max(0.0, 1.0 - sum(distances) / len(distances))


# ── Core Grading ────────────────────────────────────────────────────────────


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
    undo_count: int = 0,
    validate_count: int = 0,
    undo_cost: float = 0.02,
    validate_cost: float = 0.01,
    rules: list | None = None,
    cross_column_maps: dict | None = None,
    cached_schema_score: float | None = None,
    cached_row_mapping: dict[int, int] | None = None,
) -> tuple[dict[str, str], float, float, dict[int, int]]:
    """Grade the agent's result using multi-level scoring.

    Four scoring dimensions:
        - Schema (0.15): column name matching + type compatibility
        - Row (0.20): row count correctness, spurious/missing row handling
        - Cell (0.55): severity-weighted cell error resolution
        - Distribution (0.10): imputation quality for null-filled columns

    Returns:
        (error_status, reward, schema_score_val, row_mapping).
    """
    s_score = cached_schema_score if cached_schema_score is not None else schema_score(clean_df, result_df)
    row_map = cached_row_mapping if cached_row_mapping is not None else match_rows_by_content(clean_df, result_df)
    r_score = row_score(clean_df, result_df, error_map)

    # Identify imputed columns (those with null errors)
    imputed_cols: set[str] = set()
    for key, info in error_map.get("cell_errors", {}).items():
        if info.get("corruption") in ("inject_nulls", "null_injected"):
            try:
                _, col = _parse_error_key(key)
            except ValueError:
                continue
            imputed_cols.add(col)

    c_score, error_status = _cell_score_full(clean_df, result_df, error_map, row_map)
    d_score = distribution_score(clean_df, result_df, imputed_cols)

    # Semantic score (5th dimension)
    sem_score = _compute_semantic(result_df, rules, cross_column_maps)

    # Weights: with rules present, use 5-dimension; otherwise legacy 4-dimension
    if rules:
        constraint = s_score * 0.15 + r_score * 0.15 + c_score * 0.50 + d_score * 0.10 + sem_score * 0.10
    else:
        constraint = s_score * 0.15 + r_score * 0.20 + c_score * 0.55 + d_score * 0.10

    transform_excess = max(0, transform_steps - min_transform_steps)
    transform_penalty = transform_excess / (max_transform_steps * 2) if max_transform_steps > 0 else 0
    normal_explores = explore_steps - explore_timeouts
    explore_penalty = (
        normal_explores * explore_cost_per_step
        + explore_timeouts * explore_timeout_cost
    )
    undo_penalty = undo_count * undo_cost
    validate_penalty = validate_count * validate_cost
    efficiency = max(
        0.5, 1.0 - transform_penalty - explore_penalty - undo_penalty - validate_penalty
    )

    reward = round(min(1.0, constraint * efficiency), 4)
    return error_status, reward, s_score, row_map


def _compute_semantic(
    result_df: pd.DataFrame,
    rules: list | None,
    cross_column_maps: dict | None,
) -> float:
    """Compute semantic score. Returns 1.0 if no rules."""
    if not rules:
        return 1.0
    try:
        from server.rules.validator import compute_semantic_score
    except ImportError:
        from rules.validator import compute_semantic_score
    return compute_semantic_score(result_df, rules, cross_column_maps)


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
