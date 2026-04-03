"""
Constraint-based grader for data cleaning tasks.

Each constraint type has a checker function that evaluates whether the
constraint is satisfied on the current DataFrame. The grader aggregates
results into a score in [0.0, 1.0].
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


# ── Constraint Checkers ──────────────────────────────────────────────────────


def check_no_nulls(df: pd.DataFrame, constraint: dict[str, Any]) -> bool:
    col = constraint["column"]
    if col not in df.columns:
        return False
    return df[col].isnull().sum() == 0


def check_dtype(df: pd.DataFrame, constraint: dict[str, Any]) -> bool:
    col = constraint["column"]
    expected = constraint["dtype"]
    if col not in df.columns:
        return False
    try:
        # Try to convert and check
        converted = pd.to_numeric(df[col], errors="coerce") if "float" in expected or "int" in expected else df[col]
        if "float" in expected:
            return converted.dtype in (np.float64, np.float32, float)
        elif "int" in expected:
            # Check no NaNs (can't be int with NaN) and all values are whole numbers
            if converted.isnull().any():
                return False
            return (converted == converted.astype(int)).all()
        return str(df[col].dtype) == expected
    except Exception:
        return False


def check_no_duplicates(df: pd.DataFrame, constraint: dict[str, Any]) -> bool:
    key = constraint.get("key")
    if key is None:
        return df.duplicated().sum() == 0
    missing_cols = [k for k in key if k not in df.columns]
    if missing_cols:
        return False
    return df.duplicated(subset=key).sum() == 0


def check_value_range(df: pd.DataFrame, constraint: dict[str, Any]) -> bool:
    col = constraint["column"]
    if col not in df.columns:
        return False
    try:
        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.isnull().any():
            return False
        return bool((numeric >= constraint["min"]).all() and (numeric <= constraint["max"]).all())
    except Exception:
        return False


def check_regex_match(df: pd.DataFrame, constraint: dict[str, Any]) -> bool:
    col = constraint["column"]
    pattern = constraint["pattern"]
    if col not in df.columns:
        return False
    try:
        return bool(df[col].astype(str).str.match(pattern).all())
    except Exception:
        return False


def check_unique_values(df: pd.DataFrame, constraint: dict[str, Any]) -> bool:
    col = constraint["column"]
    allowed = set(constraint["allowed"])
    if col not in df.columns:
        return False
    # Drop NaNs before checking (NaN handling is a separate constraint)
    values = df[col].dropna().unique()
    return all(v in allowed for v in values)


def check_row_count_range(df: pd.DataFrame, constraint: dict[str, Any]) -> bool:
    return constraint["min"] <= len(df) <= constraint["max"]


def check_no_whitespace(df: pd.DataFrame, constraint: dict[str, Any]) -> bool:
    col = constraint["column"]
    if col not in df.columns:
        return False
    try:
        str_col = df[col].dropna().astype(str)
        return bool((str_col == str_col.str.strip()).all())
    except Exception:
        return False


def check_consistent_case(df: pd.DataFrame, constraint: dict[str, Any]) -> bool:
    col = constraint["column"]
    if col not in df.columns:
        return False
    try:
        values = df[col].dropna().astype(str).unique()
        if len(values) == 0:
            return True
        # Check if all values follow a single casing pattern
        all_upper = all(v == v.upper() for v in values)
        all_lower = all(v == v.lower() for v in values)
        all_title = all(v == v.title() for v in values)
        return all_upper or all_lower or all_title
    except Exception:
        return False


def check_cross_column(df: pd.DataFrame, constraint: dict[str, Any]) -> bool:
    expr = constraint["expression"]
    try:
        result = df.eval(expr)
        return bool(result.all())
    except Exception:
        return False


# ── Dispatcher ───────────────────────────────────────────────────────────────

CHECKERS = {
    "no_nulls": check_no_nulls,
    "dtype": check_dtype,
    "no_duplicates": check_no_duplicates,
    "value_range": check_value_range,
    "regex_match": check_regex_match,
    "unique_values": check_unique_values,
    "row_count_range": check_row_count_range,
    "no_whitespace": check_no_whitespace,
    "consistent_case": check_consistent_case,
    "cross_column": check_cross_column,
}


def grade(df: pd.DataFrame, constraints: list[dict[str, Any]]) -> dict[str, bool]:
    """Evaluate all constraints against the DataFrame.

    Returns a dict of {constraint_id: satisfied}.
    """
    results = {}
    for c in constraints:
        checker = CHECKERS.get(c["type"])
        if checker is None:
            results[c["id"]] = False
            continue
        try:
            results[c["id"]] = checker(df, c)
        except Exception:
            results[c["id"]] = False
    return results


def compute_reward(
    constraint_results: dict[str, bool],
    constraints: list[dict[str, Any]],
    transform_steps: int,
    min_transform_steps: int,
    max_transform_steps: int,
) -> float:
    """Compute final reward from severity-weighted constraint satisfaction + efficiency.

    score = Σ(severity_i * solved_i) / Σ(severity_i)   [0.0–1.0]
    reward = score * efficiency_factor

    Severity levels:
      critical=3, high=2, medium=1 (default=1)
    """
    if not constraint_results:
        return 0.0

    severity_map = {"critical": 3, "high": 2, "medium": 1}

    weighted_solved = 0.0
    weighted_total = 0.0
    for c in constraints:
        sev = severity_map.get(c.get("severity", "medium"), 1)
        weighted_total += sev
        if constraint_results.get(c["id"], False):
            weighted_solved += sev

    if weighted_total == 0:
        return 0.0

    constraint_score = weighted_solved / weighted_total

    excess_steps = max(0, transform_steps - min_transform_steps)
    efficiency_factor = max(0.5, 1.0 - excess_steps / (max_transform_steps * 2))

    return round(constraint_score * efficiency_factor, 4)
