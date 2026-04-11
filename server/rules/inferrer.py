"""Auto-infer semantic rules from clean DataFrames."""
from __future__ import annotations

from typing import Optional

import pandas as pd

try:
    from server.rules.types import (
        CrossColumnRule,
        EnumRule,
        NotNullRule,
        RangeRule,
        RegexRule,
        Rule,
        UniqueRule,
    )
except ImportError:
    from rules.types import (
        CrossColumnRule,
        EnumRule,
        NotNullRule,
        RangeRule,
        RegexRule,
        Rule,
        UniqueRule,
    )


_MAX_ENUM_CARDINALITY = 30
_MIN_FD_ROWS = 3

_RANGE_HEURISTICS: dict[tuple[str, ...], tuple[float, float]] = {
    ("age", "years", "year_of_age"): (0.0, 120.0),
    ("salary", "income", "wage", "wages", "pay", "earnings"): (0.0, float("inf")),
    ("price", "cost", "amount", "total", "fee", "charge"): (0.0, float("inf")),
    ("rating",): (0.0, 10.0),
    ("percentage", "pct", "percent"): (0.0, 100.0),
}

_REGEX_HEURISTICS: dict[tuple[str, ...], str] = {
    ("email", "email_address", "e_mail"): r"^[\w.+-]+@[\w-]+\.[\w.]+$",
    ("zip", "zipcode", "zip_code", "postal_code", "pincode", "pin_code"): r"^\d{4,6}$",
    ("phone", "phone_number", "telephone", "tel"): r"^[\d\s\-\+\(\)]{7,20}$",
    ("url", "website", "homepage", "link"): r"^https?://",
    ("ssn", "social_security"): r"^\d{3}-?\d{2}-?\d{4}$",
}

_ID_PATTERNS: tuple[str, ...] = (
    "_id",
    "id_",
    "identifier",
    "key",
)


def infer_rules(df: pd.DataFrame, domain: Optional[str] = None) -> list[Rule]:
    """Infer semantic rules from a clean DataFrame.

    The optional ``domain`` argument is accepted for future domain-specific
    heuristics; current inference is based on column names and observed data.
    """
    del domain

    rules: list[Rule] = []
    heuristic_rule_types_by_col: dict[str, set[str]] = {}

    for col in df.columns:
        col_rules = _infer_heuristic(col)
        if not col_rules:
            continue

        rules.extend(col_rules)
        heuristic_rule_types_by_col[col] = {rule.rule_type for rule in col_rules}

    for col in df.columns:
        rules.extend(
            _infer_statistical(
                col,
                df[col],
                suppressed_rule_types=heuristic_rule_types_by_col.get(col, set()),
            )
        )

    rules.extend(_infer_cross_column(df))
    return rules


def _infer_heuristic(col: str) -> list[Rule]:
    rules: list[Rule] = []

    for patterns, (min_val, max_val) in _RANGE_HEURISTICS.items():
        if _matches_heuristic(col, patterns):
            rules.append(
                RangeRule(
                    column=col,
                    min_val=min_val,
                    max_val=max_val,
                    source="heuristic",
                )
            )
            break

    for patterns, pattern in _REGEX_HEURISTICS.items():
        if _matches_heuristic(col, patterns):
            rules.append(RegexRule(column=col, pattern=pattern, source="heuristic"))
            break

    if _is_id_column(col):
        rules.append(UniqueRule(column=col, source="heuristic"))
        rules.append(NotNullRule(column=col, source="heuristic"))

    return rules


def _infer_statistical(
    col: str,
    series: pd.Series,
    suppressed_rule_types: set[str],
) -> list[Rule]:
    rules: list[Rule] = []

    if "not_null" not in suppressed_rule_types and series.notna().all():
        rules.append(NotNullRule(column=col, source="statistical"))

    if "range" not in suppressed_rule_types and _is_numeric_like(series):
        numeric = pd.to_numeric(series.dropna(), errors="coerce").dropna()
        if not numeric.empty:
            min_val, max_val = _expanded_numeric_range(numeric)
            rules.append(
                RangeRule(
                    column=col,
                    min_val=min_val,
                    max_val=max_val,
                    source="statistical",
                )
            )

    if "enum" not in suppressed_rule_types and _is_categorical_like(series):
        non_null = series.dropna()
        unique_vals = non_null.unique()
        if 1 < len(unique_vals) <= _MAX_ENUM_CARDINALITY:
            values = sorted(str(value) for value in unique_vals)
            rules.append(EnumRule(column=col, values=values, source="statistical"))

    return rules


def _expanded_numeric_range(numeric: pd.Series) -> tuple[float, float]:
    observed_min = float(numeric.min())
    observed_max = float(numeric.max())
    spread = observed_max - observed_min
    if spread == 0:
        spread = abs(observed_max) if observed_max != 0 else 1.0

    expanded_min = observed_min - (0.2 * spread)
    expanded_max = observed_max + (0.2 * spread)
    if observed_min >= 0:
        expanded_min = 0.0

    return expanded_min, expanded_max


def _infer_cross_column(df: pd.DataFrame) -> list[Rule]:
    rules: list[Rule] = []
    categorical_cols = [
        col
        for col in df.columns
        if _is_categorical_like(df[col])
        and 1 < df[col].dropna().nunique() <= _MAX_ENUM_CARDINALITY
    ]

    for index, col_a in enumerate(categorical_cols):
        for col_b in categorical_cols[index + 1 :]:
            if _is_functional_dependency(df, col_a, col_b):
                rules.append(_functional_dependency_rule(df, col_a, col_b))
            elif _is_functional_dependency(df, col_b, col_a):
                rules.append(_functional_dependency_rule(df, col_b, col_a))

    return rules


def _functional_dependency_rule(
    df: pd.DataFrame,
    determinant: str,
    dependent: str,
) -> CrossColumnRule:
    mapping = _describe_mapping(df, determinant, dependent)
    return CrossColumnRule(
        columns=[determinant, dependent],
        condition="functional_dependency",
        hint=(
            f"Each {determinant} value maps to exactly one "
            f"{dependent} value: {mapping}"
        ),
        source="cross_column_inference",
    )


def _is_functional_dependency(df: pd.DataFrame, col_a: str, col_b: str) -> bool:
    pairs = df[[col_a, col_b]].dropna()
    if len(pairs) < _MIN_FD_ROWS:
        return False

    grouped = pairs.groupby(col_a, dropna=True)[col_b].nunique()
    if grouped.empty or not bool((grouped == 1).all()):
        return False

    # Avoid inferring arbitrary dependencies from columns where every value
    # appears only once in the sample.
    return bool(pairs[col_a].duplicated(keep=False).any())


def _describe_mapping(
    df: pd.DataFrame,
    col_a: str,
    col_b: str,
    max_entries: int = 8,
) -> str:
    pairs = df[[col_a, col_b]].dropna().drop_duplicates()
    mapping = dict(zip(pairs[col_a].astype(str), pairs[col_b].astype(str)))
    items = list(mapping.items())[:max_entries]
    suffix = f" (+{len(mapping) - max_entries} more)" if len(mapping) > max_entries else ""
    return ", ".join(f"{key}={value}" for key, value in items) + suffix


def _matches_heuristic(col: str, patterns: tuple[str, ...]) -> bool:
    normalized = _normalize_column_name(col)
    for pattern in patterns:
        normalized_pattern = _normalize_column_name(pattern)
        if (
            normalized == normalized_pattern
            or normalized.endswith(f"_{normalized_pattern}")
            or normalized.startswith(f"{normalized_pattern}_")
        ):
            return True
    return False


def _is_id_column(col: str) -> bool:
    normalized = _normalize_column_name(col)
    if normalized == "id" or normalized.endswith("_id"):
        return True
    return any(pattern in normalized for pattern in _ID_PATTERNS)


def _normalize_column_name(col: str) -> str:
    return str(col).strip().lower()


def _is_numeric_like(series: pd.Series) -> bool:
    non_null = series.dropna()
    if non_null.empty:
        return False

    numeric = pd.to_numeric(non_null, errors="coerce")
    return bool(numeric.notna().mean() > 0.8)


def _is_categorical_like(series: pd.Series) -> bool:
    return (
        pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series)
        or isinstance(series.dtype, pd.CategoricalDtype)
    )
