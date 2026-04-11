"""Validate a DataFrame against semantic rules."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd

try:
    from server.rules.types import (
        CrossColumnRule,
        DtypeRule,
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
        DtypeRule,
        EnumRule,
        NotNullRule,
        RangeRule,
        RegexRule,
        Rule,
        UniqueRule,
    )


@dataclass
class Violation:
    """A single rule violation found in a DataFrame."""

    rule_type: str
    column: str
    row_index: int
    expected: str
    actual_value: Any
    rule: Rule


def validate(
    df: pd.DataFrame,
    rules: list[Rule],
    cross_column_maps: Optional[dict[str, dict]] = None,
) -> list[Violation]:
    """Check *df* against every rule and return detected violations."""
    violations: list[Violation] = []
    maps = cross_column_maps or {}

    for rule in rules:
        if isinstance(rule, CrossColumnRule):
            violations.extend(_check_cross_column(df, rule, maps))
        elif hasattr(rule, "column") and rule.column not in df.columns:
            continue
        elif isinstance(rule, RangeRule):
            violations.extend(_check_range(df, rule))
        elif isinstance(rule, RegexRule):
            violations.extend(_check_regex(df, rule))
        elif isinstance(rule, EnumRule):
            violations.extend(_check_enum(df, rule))
        elif isinstance(rule, DtypeRule):
            violations.extend(_check_dtype(df, rule))
        elif isinstance(rule, NotNullRule):
            violations.extend(_check_not_null(df, rule))
        elif isinstance(rule, UniqueRule):
            violations.extend(_check_unique(df, rule))

    return violations


def compute_semantic_score(
    df: pd.DataFrame,
    rules: list[Rule],
    cross_column_maps: Optional[dict[str, dict]] = None,
) -> float:
    """Compute semantic validity score in [0.0, 1.0]."""
    if not rules:
        return 1.0

    violations = validate(df, rules, cross_column_maps)
    total_checks = 0

    for rule in rules:
        if isinstance(rule, CrossColumnRule):
            if all(column in df.columns for column in rule.columns):
                total_checks += len(df)
        elif isinstance(rule, UniqueRule):
            if rule.column in df.columns:
                total_checks += len(df)
        elif hasattr(rule, "column") and rule.column in df.columns:
            total_checks += len(df)

    if total_checks == 0:
        return 1.0

    return max(0.0, 1.0 - len(violations) / total_checks)


def _check_range(df: pd.DataFrame, rule: RangeRule) -> list[Violation]:
    col = df[rule.column]
    numeric = pd.to_numeric(col, errors="coerce")
    violations: list[Violation] = []

    for pos, row_index in enumerate(df.index):
        val = numeric.iloc[pos]
        if pd.isna(val):
            continue
        if val < rule.min_val or val > rule.max_val:
            violations.append(
                Violation(
                    rule_type="range",
                    column=rule.column,
                    row_index=row_index,
                    expected=f"[{rule.min_val}, {rule.max_val}]",
                    actual_value=col.iloc[pos],
                    rule=rule,
                )
            )

    return violations


def _check_regex(df: pd.DataFrame, rule: RegexRule) -> list[Violation]:
    col = df[rule.column]
    pattern = re.compile(rule.pattern)
    violations: list[Violation] = []

    for pos, row_index in enumerate(df.index):
        raw_val = col.iloc[pos]
        if pd.isna(raw_val):
            continue
        val = str(raw_val)
        if not pattern.search(val):
            violations.append(
                Violation(
                    rule_type="regex",
                    column=rule.column,
                    row_index=row_index,
                    expected=f"matches {rule.pattern}",
                    actual_value=val,
                    rule=rule,
                )
            )

    return violations


def _check_enum(df: pd.DataFrame, rule: EnumRule) -> list[Violation]:
    col = df[rule.column]
    allowed = set(rule.values)
    allowed_lower = {str(value).lower() for value in allowed}
    violations: list[Violation] = []

    for pos, row_index in enumerate(df.index):
        val = col.iloc[pos]
        if pd.isna(val):
            continue
        if str(val).lower() not in allowed_lower:
            violations.append(
                Violation(
                    rule_type="enum",
                    column=rule.column,
                    row_index=row_index,
                    expected=f"one of {sorted(allowed)}",
                    actual_value=val,
                    rule=rule,
                )
            )

    return violations


def _check_dtype(df: pd.DataFrame, rule: DtypeRule) -> list[Violation]:
    col = df[rule.column]
    violations: list[Violation] = []

    for pos, row_index in enumerate(df.index):
        val = col.iloc[pos]
        if pd.isna(val):
            continue
        if not _is_compatible_dtype(val, rule.expected_dtype):
            violations.append(
                Violation(
                    rule_type="dtype",
                    column=rule.column,
                    row_index=row_index,
                    expected=f"dtype {rule.expected_dtype}",
                    actual_value=val,
                    rule=rule,
                )
            )

    return violations


def _is_compatible_dtype(val: Any, expected: str) -> bool:
    if expected == "integer":
        try:
            numeric = float(val)
            return numeric == int(numeric)
        except (TypeError, ValueError):
            return False
    if expected == "float":
        try:
            float(val)
            return True
        except (TypeError, ValueError):
            return False
    if expected == "date":
        try:
            pd.Timestamp(val)
            return True
        except (TypeError, ValueError):
            return False
    if expected == "string":
        return isinstance(val, str)
    return True


def _check_not_null(df: pd.DataFrame, rule: NotNullRule) -> list[Violation]:
    col = df[rule.column]
    violations: list[Violation] = []

    for pos, row_index in enumerate(df.index):
        if pd.isna(col.iloc[pos]):
            violations.append(
                Violation(
                    rule_type="not_null",
                    column=rule.column,
                    row_index=row_index,
                    expected="not null",
                    actual_value=None,
                    rule=rule,
                )
            )

    return violations


def _check_unique(df: pd.DataFrame, rule: UniqueRule) -> list[Violation]:
    col = df[rule.column]
    duplicated_mask = col.duplicated(keep=False)
    violations: list[Violation] = []
    seen_vals: set[Any] = set()

    for pos, row_index in enumerate(df.index):
        val = col.iloc[pos]
        if pd.isna(val):
            continue
        if duplicated_mask.iloc[pos] and val in seen_vals:
            violations.append(
                Violation(
                    rule_type="unique",
                    column=rule.column,
                    row_index=row_index,
                    expected="unique value",
                    actual_value=val,
                    rule=rule,
                )
            )
        seen_vals.add(val)

    return violations


def _check_cross_column(
    df: pd.DataFrame,
    rule: CrossColumnRule,
    maps: dict[str, dict],
) -> list[Violation]:
    if not all(column in df.columns for column in rule.columns):
        return []

    violations: list[Violation] = []

    if rule.condition == "functional_dependency" and len(rule.columns) == 2:
        col_a, col_b = rule.columns
        mapping = maps.get(f"{col_a}->{col_b}", {})
        if not mapping:
            return []

        for pos, row_index in enumerate(df.index):
            val_a = df[col_a].iloc[pos]
            val_b = df[col_b].iloc[pos]
            if pd.isna(val_a) or pd.isna(val_b):
                continue

            expected_b = mapping.get(str(val_a))
            if expected_b is not None and str(val_b) != str(expected_b):
                violations.append(
                    Violation(
                        rule_type="cross_column",
                        column=f"{col_a},{col_b}",
                        row_index=row_index,
                        expected=f"{col_a}={val_a} -> {col_b}={expected_b}",
                        actual_value=val_b,
                        rule=rule,
                    )
                )
    elif rule.condition == "ordering" and len(rule.columns) == 2:
        col_a, col_b = rule.columns

        for pos, row_index in enumerate(df.index):
            val_a = pd.to_numeric(df[col_a].iloc[pos], errors="coerce")
            val_b = pd.to_numeric(df[col_b].iloc[pos], errors="coerce")
            if pd.isna(val_a) or pd.isna(val_b):
                continue
            if val_a > val_b:
                violations.append(
                    Violation(
                        rule_type="cross_column",
                        column=f"{col_a},{col_b}",
                        row_index=row_index,
                        expected=f"{col_a} <= {col_b}",
                        actual_value=f"{val_a} > {val_b}",
                        rule=rule,
                    )
                )

    return violations
