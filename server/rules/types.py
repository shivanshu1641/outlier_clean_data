"""Semantic rule types for data validation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Union


@dataclass(frozen=True)
class RangeRule:
    """Numeric column values must fall within an inclusive range."""

    column: str
    min_val: float
    max_val: float
    source: str = "inferred"
    rule_type: str = field(default="range", init=False)


@dataclass(frozen=True)
class RegexRule:
    """String column values must match the regex pattern."""

    column: str
    pattern: str
    source: str = "inferred"
    rule_type: str = field(default="regex", init=False)


@dataclass(frozen=True)
class EnumRule:
    """Column values must be one of the allowed values."""

    column: str
    values: list[str]
    source: str = "inferred"
    rule_type: str = field(default="enum", init=False)


@dataclass(frozen=True)
class DtypeRule:
    """Column values must be parseable as the expected dtype."""

    column: str
    expected_dtype: str
    source: str = "inferred"
    rule_type: str = field(default="dtype", init=False)


@dataclass(frozen=True)
class NotNullRule:
    """Column values must not be null."""

    column: str
    source: str = "inferred"
    rule_type: str = field(default="not_null", init=False)


@dataclass(frozen=True)
class UniqueRule:
    """Column values must be unique."""

    column: str
    source: str = "inferred"
    rule_type: str = field(default="unique", init=False)


@dataclass(frozen=True)
class CrossColumnRule:
    """Relationship between two or more columns must hold."""

    columns: list[str]
    condition: str
    hint: str
    source: str = "inferred"
    rule_type: str = field(default="cross_column", init=False)


Rule = Union[
    RangeRule,
    RegexRule,
    EnumRule,
    DtypeRule,
    NotNullRule,
    UniqueRule,
    CrossColumnRule,
]

_TYPE_MAP: dict[str, type[Rule]] = {
    "range": RangeRule,
    "regex": RegexRule,
    "enum": EnumRule,
    "dtype": DtypeRule,
    "not_null": NotNullRule,
    "unique": UniqueRule,
    "cross_column": CrossColumnRule,
}


def rule_to_dict(rule: Rule) -> dict[str, Any]:
    """Serialize a rule to a JSON-compatible dictionary."""
    result: dict[str, Any] = {
        "type": rule.rule_type,
        "source": rule.source,
    }

    if isinstance(rule, RangeRule):
        result.update({
            "column": rule.column,
            "min": rule.min_val,
            "max": rule.max_val,
        })
    elif isinstance(rule, RegexRule):
        result.update({
            "column": rule.column,
            "pattern": rule.pattern,
        })
    elif isinstance(rule, EnumRule):
        result.update({
            "column": rule.column,
            "values": list(rule.values),
        })
    elif isinstance(rule, DtypeRule):
        result.update({
            "column": rule.column,
            "expected_dtype": rule.expected_dtype,
        })
    elif isinstance(rule, NotNullRule):
        result["column"] = rule.column
    elif isinstance(rule, UniqueRule):
        result["column"] = rule.column
    elif isinstance(rule, CrossColumnRule):
        result.update({
            "columns": list(rule.columns),
            "condition": rule.condition,
            "hint": rule.hint,
        })
    else:
        raise ValueError(f"Unknown rule instance: {rule!r}")

    return result


def rule_from_dict(data: dict[str, Any]) -> Rule:
    """Deserialize a JSON-compatible dictionary into a rule."""
    rule_type = data.get("type")
    if rule_type not in _TYPE_MAP:
        raise ValueError(f"Unknown rule type: {rule_type!r}")

    source = data.get("source", "inferred")

    if rule_type == "range":
        return RangeRule(
            column=data["column"],
            min_val=data["min"],
            max_val=data["max"],
            source=source,
        )
    if rule_type == "regex":
        return RegexRule(
            column=data["column"],
            pattern=data["pattern"],
            source=source,
        )
    if rule_type == "enum":
        return EnumRule(
            column=data["column"],
            values=list(data["values"]),
            source=source,
        )
    if rule_type == "dtype":
        return DtypeRule(
            column=data["column"],
            expected_dtype=data["expected_dtype"],
            source=source,
        )
    if rule_type == "not_null":
        return NotNullRule(column=data["column"], source=source)
    if rule_type == "unique":
        return UniqueRule(column=data["column"], source=source)
    if rule_type == "cross_column":
        return CrossColumnRule(
            columns=list(data["columns"]),
            condition=data["condition"],
            hint=data["hint"],
            source=source,
        )

    raise ValueError(f"Unknown rule type: {rule_type!r}")
