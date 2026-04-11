# Semantic Constraints + Category System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add domain-aware semantic rules (auto-inferred + manual) as a 5th grading dimension, and a category system that constrains corruption selection for targeted benchmarking.

**Architecture:** New `server/rules/` package owns rule types, inference, and validation. Categories live in `server/corruption/categories.py` and constrain what the pipeline selects. Both integrate into the existing generative pipeline at `reset()` time — rules are data in `catalog.json`, not code.

**Tech Stack:** Python 3.11, pandas, numpy, pytest. No new dependencies.

---

## File Structure

### New Files
| File | Responsibility |
|------|----------------|
| `server/rules/__init__.py` | Package exports |
| `server/rules/types.py` | 7 rule dataclasses + JSON serialization |
| `server/rules/inferrer.py` | 3-layer auto-inference engine |
| `server/rules/validator.py` | Validate DataFrame against ruleset |
| `server/rules/catalog_enricher.py` | CLI to enrich catalog.json with inferred rules |
| `server/corruption/categories.py` | Category definitions + corruption/format maps |
| `tests/test_rules.py` | Tests for types, inferrer, validator |
| `tests/test_categories.py` | Tests for category mappings + constrained generation |
| `tests/test_grader_semantic.py` | Tests for 5th grading dimension |

### Modified Files
| File | What Changes |
|------|-------------|
| `server/grader.py:588-662` | Add `semantic_score()`, rebalance weights in `grade()` |
| `server/corruption/pipeline.py:42-164` | Accept `rules` + `category` params |
| `server/corruption/value_corruptions.py:842-890` | Rewrite `business_rule_violation` to use rule types |
| `server/corruption/profiles.py:4-33` | No structural change — categories override format_pool |
| `server/corruption/__init__.py` | Export `CATEGORY_CORRUPTION_MAP`, `CATEGORY_FORMAT_MAP` |
| `server/environment.py:230-327` | Pass `category` + `rules` through reset/pipeline |
| `models.py:151-166` | Add `semantic_rules` field to `DataCleaningObservation` |
| `datasets/catalog.json` | Add `rules` field per dataset (auto-generated) |

---

## Task 1: Rule Type Definitions

**Files:**
- Create: `server/rules/__init__.py`
- Create: `server/rules/types.py`
- Test: `tests/test_rules.py`

- [ ] **Step 1: Write failing tests for rule types**

```python
# tests/test_rules.py
import pytest
from server.rules.types import (
    RangeRule, RegexRule, EnumRule, DtypeRule,
    NotNullRule, UniqueRule, CrossColumnRule,
    rule_from_dict, rule_to_dict,
)


class TestRuleTypes:
    def test_range_rule_creation(self):
        r = RangeRule(column="age", min_val=0, max_val=120, source="heuristic")
        assert r.column == "age"
        assert r.min_val == 0
        assert r.max_val == 120
        assert r.rule_type == "range"

    def test_regex_rule_creation(self):
        r = RegexRule(column="email", pattern=r"^[\w.]+@[\w.]+\.\w+$", source="heuristic")
        assert r.rule_type == "regex"
        assert r.pattern == r"^[\w.]+@[\w.]+\.\w+$"

    def test_enum_rule_creation(self):
        r = EnumRule(column="sex", values=["male", "female"], source="statistical")
        assert r.rule_type == "enum"
        assert set(r.values) == {"male", "female"}

    def test_dtype_rule_creation(self):
        r = DtypeRule(column="age", expected_dtype="integer", source="statistical")
        assert r.rule_type == "dtype"

    def test_not_null_rule_creation(self):
        r = NotNullRule(column="id", source="statistical")
        assert r.rule_type == "not_null"

    def test_unique_rule_creation(self):
        r = UniqueRule(column="id", source="heuristic")
        assert r.rule_type == "unique"

    def test_cross_column_rule_creation(self):
        r = CrossColumnRule(
            columns=["embarked", "port"],
            condition="functional_dependency",
            hint="S=Southampton, C=Cherbourg",
            source="cross_column_inference",
        )
        assert r.rule_type == "cross_column"
        assert r.hint == "S=Southampton, C=Cherbourg"


class TestRuleSerialization:
    def test_round_trip_range(self):
        original = RangeRule(column="age", min_val=0, max_val=120, source="manual")
        d = rule_to_dict(original)
        assert d["type"] == "range"
        assert d["column"] == "age"
        restored = rule_from_dict(d)
        assert isinstance(restored, RangeRule)
        assert restored.min_val == 0
        assert restored.max_val == 120
        assert restored.source == "manual"

    def test_round_trip_enum(self):
        original = EnumRule(column="sex", values=["male", "female"], source="statistical")
        d = rule_to_dict(original)
        restored = rule_from_dict(d)
        assert isinstance(restored, EnumRule)
        assert restored.values == ["male", "female"]

    def test_round_trip_cross_column(self):
        original = CrossColumnRule(
            columns=["city", "state"],
            condition="functional_dependency",
            hint="City determines state",
            source="cross_column_inference",
        )
        d = rule_to_dict(original)
        assert d["columns"] == ["city", "state"]
        restored = rule_from_dict(d)
        assert isinstance(restored, CrossColumnRule)
        assert restored.hint == "City determines state"

    def test_from_dict_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown rule type"):
            rule_from_dict({"type": "bogus", "column": "x"})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/shivanshu.s/deployments/proj/outlier_clean_data/.claude/worktrees/env-v2 && python -m pytest tests/test_rules.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'server.rules'`

- [ ] **Step 3: Implement rule types**

```python
# server/rules/__init__.py
from .types import (
    RangeRule,
    RegexRule,
    EnumRule,
    DtypeRule,
    NotNullRule,
    UniqueRule,
    CrossColumnRule,
    Rule,
    rule_from_dict,
    rule_to_dict,
)

__all__ = [
    "RangeRule",
    "RegexRule",
    "EnumRule",
    "DtypeRule",
    "NotNullRule",
    "UniqueRule",
    "CrossColumnRule",
    "Rule",
    "rule_from_dict",
    "rule_to_dict",
]
```

```python
# server/rules/types.py
"""Semantic rule types for data validation.

Seven rule types, all JSON-serializable. Each rule validates one aspect
of a DataFrame column (or column pair for cross-column rules).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Union


@dataclass(frozen=True)
class RangeRule:
    """Numeric column must fall within [min_val, max_val]."""
    column: str
    min_val: float
    max_val: float
    source: str = "inferred"
    rule_type: str = field(default="range", init=False)


@dataclass(frozen=True)
class RegexRule:
    """String column must match the given regex pattern."""
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
    """Column must be parseable as the expected dtype (integer, float, date, string)."""
    column: str
    expected_dtype: str  # "integer", "float", "date", "string"
    source: str = "inferred"
    rule_type: str = field(default="dtype", init=False)


@dataclass(frozen=True)
class NotNullRule:
    """Column must not contain null/NaN values."""
    column: str
    source: str = "inferred"
    rule_type: str = field(default="not_null", init=False)


@dataclass(frozen=True)
class UniqueRule:
    """Column values must be unique (no duplicates)."""
    column: str
    source: str = "inferred"
    rule_type: str = field(default="unique", init=False)


@dataclass(frozen=True)
class CrossColumnRule:
    """Relationship between two or more columns must hold."""
    columns: list[str]
    condition: str  # "functional_dependency", "ordering", etc.
    hint: str  # Natural-language description for the agent
    source: str = "inferred"
    rule_type: str = field(default="cross_column", init=False)


# Union of all rule types
Rule = Union[RangeRule, RegexRule, EnumRule, DtypeRule, NotNullRule, UniqueRule, CrossColumnRule]

_TYPE_MAP: dict[str, type] = {
    "range": RangeRule,
    "regex": RegexRule,
    "enum": EnumRule,
    "dtype": DtypeRule,
    "not_null": NotNullRule,
    "unique": UniqueRule,
    "cross_column": CrossColumnRule,
}

# Fields that are NOT constructor arguments (handled separately)
_SKIP_FIELDS = {"rule_type", "type"}


def rule_to_dict(rule: Rule) -> dict[str, Any]:
    """Serialize a rule to a JSON-compatible dict."""
    d: dict[str, Any] = {"type": rule.rule_type}
    if hasattr(rule, "column"):
        d["column"] = rule.column
    if hasattr(rule, "columns"):
        d["columns"] = list(rule.columns)
    # Add type-specific fields
    if isinstance(rule, RangeRule):
        d["min"] = rule.min_val
        d["max"] = rule.max_val
    elif isinstance(rule, RegexRule):
        d["pattern"] = rule.pattern
    elif isinstance(rule, EnumRule):
        d["values"] = list(rule.values)
    elif isinstance(rule, DtypeRule):
        d["expected_dtype"] = rule.expected_dtype
    elif isinstance(rule, CrossColumnRule):
        d["condition"] = rule.condition
        d["hint"] = rule.hint
    d["source"] = rule.source
    return d


def rule_from_dict(d: dict[str, Any]) -> Rule:
    """Deserialize a dict into the appropriate Rule type."""
    rule_type = d.get("type")
    if rule_type not in _TYPE_MAP:
        raise ValueError(f"Unknown rule type: {rule_type!r}")

    cls = _TYPE_MAP[rule_type]
    kwargs: dict[str, Any] = {"source": d.get("source", "inferred")}

    if rule_type == "range":
        kwargs["column"] = d["column"]
        kwargs["min_val"] = d["min"]
        kwargs["max_val"] = d["max"]
    elif rule_type == "regex":
        kwargs["column"] = d["column"]
        kwargs["pattern"] = d["pattern"]
    elif rule_type == "enum":
        kwargs["column"] = d["column"]
        kwargs["values"] = d["values"]
    elif rule_type == "dtype":
        kwargs["column"] = d["column"]
        kwargs["expected_dtype"] = d["expected_dtype"]
    elif rule_type == "not_null":
        kwargs["column"] = d["column"]
    elif rule_type == "unique":
        kwargs["column"] = d["column"]
    elif rule_type == "cross_column":
        kwargs["columns"] = d["columns"]
        kwargs["condition"] = d["condition"]
        kwargs["hint"] = d["hint"]

    return cls(**kwargs)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/shivanshu.s/deployments/proj/outlier_clean_data/.claude/worktrees/env-v2 && python -m pytest tests/test_rules.py -v`
Expected: All 10 tests PASS

- [ ] **Step 5: Commit**

```bash
git add server/rules/__init__.py server/rules/types.py tests/test_rules.py
git commit -m "feat(rules): rule type dataclasses with JSON serialization"
```

---

## Task 2: Rule Validator

**Files:**
- Create: `server/rules/validator.py`
- Test: `tests/test_rules.py` (append)

- [ ] **Step 1: Write failing tests for validator**

Append to `tests/test_rules.py`:

```python
import re
import pandas as pd
from server.rules.validator import validate, Violation


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "age": [25, 35, -5, 200, 45],      # -5 and 200 violate range [0, 120]
        "email": ["a@b.com", "bad", "c@d.org", "e@f.io", ""],  # "bad" and "" violate regex
        "sex": ["male", "female", "male", "alien", "female"],   # "alien" violates enum
        "score": [88.5, None, 75.0, 90.0, None],   # 2 nulls violate not_null
    })


@pytest.fixture
def sample_rules():
    return [
        RangeRule(column="age", min_val=0, max_val=120, source="heuristic"),
        RegexRule(column="email", pattern=r"^[\w.]+@[\w.]+\.\w+$", source="heuristic"),
        EnumRule(column="sex", values=["male", "female", "other"], source="statistical"),
        NotNullRule(column="score", source="statistical"),
    ]


class TestValidator:
    def test_validates_range_violations(self, sample_df, sample_rules):
        violations = validate(sample_df, [sample_rules[0]])
        assert len(violations) == 2
        assert all(v.rule_type == "range" for v in violations)
        rows = {v.row_index for v in violations}
        assert rows == {2, 3}

    def test_validates_regex_violations(self, sample_df, sample_rules):
        violations = validate(sample_df, [sample_rules[1]])
        assert len(violations) == 2
        assert all(v.rule_type == "regex" for v in violations)

    def test_validates_enum_violations(self, sample_df, sample_rules):
        violations = validate(sample_df, [sample_rules[2]])
        assert len(violations) == 1
        assert violations[0].row_index == 3
        assert violations[0].actual_value == "alien"

    def test_validates_not_null_violations(self, sample_df, sample_rules):
        violations = validate(sample_df, [sample_rules[3]])
        assert len(violations) == 2

    def test_validates_unique_rule(self):
        df = pd.DataFrame({"id": [1, 2, 3, 2, 5]})
        violations = validate(df, [UniqueRule(column="id", source="heuristic")])
        assert len(violations) >= 1  # at least the duplicate pair

    def test_validates_dtype_rule(self):
        df = pd.DataFrame({"age": [25, "thirty", 40, 50]})
        violations = validate(df, [DtypeRule(column="age", expected_dtype="integer", source="statistical")])
        assert len(violations) >= 1

    def test_all_rules_combined(self, sample_df, sample_rules):
        violations = validate(sample_df, sample_rules)
        assert len(violations) == 7  # 2 range + 2 regex + 1 enum + 2 not_null

    def test_skips_missing_column(self, sample_df):
        rule = RangeRule(column="nonexistent", min_val=0, max_val=100, source="heuristic")
        violations = validate(sample_df, [rule])
        assert len(violations) == 0

    def test_empty_rules_returns_empty(self, sample_df):
        violations = validate(sample_df, [])
        assert len(violations) == 0

    def test_semantic_score_perfect(self):
        df = pd.DataFrame({"age": [25, 35, 45]})
        rules = [RangeRule(column="age", min_val=0, max_val=120, source="heuristic")]
        violations = validate(df, rules)
        assert len(violations) == 0

    def test_cross_column_functional_dependency(self):
        df = pd.DataFrame({
            "embarked": ["S", "C", "S", "Q", "S"],
            "port": ["Southampton", "Cherbourg", "WRONG", "Queenstown", "Southampton"],
        })
        mapping = {"S": "Southampton", "C": "Cherbourg", "Q": "Queenstown"}
        rule = CrossColumnRule(
            columns=["embarked", "port"],
            condition="functional_dependency",
            hint="S=Southampton, C=Cherbourg, Q=Queenstown",
            source="manual",
            )
        violations = validate(df, [rule], cross_column_maps={"embarked->port": mapping})
        assert len(violations) == 1
        assert violations[0].row_index == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/shivanshu.s/deployments/proj/outlier_clean_data/.claude/worktrees/env-v2 && python -m pytest tests/test_rules.py::TestValidator -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'server.rules.validator'`

- [ ] **Step 3: Implement validator**

```python
# server/rules/validator.py
"""Validate a DataFrame against a list of semantic rules."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

from .types import (
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
    """A single rule violation found in the DataFrame."""
    rule_type: str
    column: str
    row_index: int
    expected: str  # Human-readable expected value/constraint
    actual_value: Any
    rule: Rule


def validate(
    df: pd.DataFrame,
    rules: list[Rule],
    cross_column_maps: Optional[dict[str, dict]] = None,
) -> list[Violation]:
    """Check *df* against every rule. Return list of violations.

    Args:
        df: DataFrame to validate.
        rules: List of Rule objects.
        cross_column_maps: Optional dict mapping "col_a->col_b" to
            {value_a: expected_value_b} for functional dependency checks.
    """
    violations: list[Violation] = []
    for rule in rules:
        if isinstance(rule, CrossColumnRule):
            violations.extend(_check_cross_column(df, rule, cross_column_maps or {}))
        elif hasattr(rule, "column") and rule.column not in df.columns:
            continue  # skip rules for columns not in the result
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
    """Compute semantic validity score in [0.0, 1.0].

    Score = 1.0 - (violating_cells / total_checkable_cells).
    If no rules apply, returns 1.0.
    """
    if not rules:
        return 1.0

    violations = validate(df, rules, cross_column_maps)
    # Count total checkable cells: one per (rule, applicable row)
    total_checks = 0
    for rule in rules:
        if isinstance(rule, CrossColumnRule):
            # Check if all columns exist
            if all(c in df.columns for c in rule.columns):
                total_checks += len(df)
        elif isinstance(rule, UniqueRule):
            if rule.column in df.columns:
                total_checks += len(df)
        elif hasattr(rule, "column") and rule.column in df.columns:
            total_checks += len(df)

    if total_checks == 0:
        return 1.0

    return max(0.0, 1.0 - len(violations) / total_checks)


# ── Individual rule checkers ──────────────────────────────────────────────


def _check_range(df: pd.DataFrame, rule: RangeRule) -> list[Violation]:
    col = df[rule.column]
    numeric = pd.to_numeric(col, errors="coerce")
    violations = []
    for idx in df.index:
        val = numeric.iloc[idx] if idx < len(numeric) else np.nan
        if pd.isna(val):
            continue  # NaN handled by NotNullRule, not range
        if val < rule.min_val or val > rule.max_val:
            violations.append(Violation(
                rule_type="range",
                column=rule.column,
                row_index=idx,
                expected=f"[{rule.min_val}, {rule.max_val}]",
                actual_value=col.iloc[idx],
                rule=rule,
            ))
    return violations


def _check_regex(df: pd.DataFrame, rule: RegexRule) -> list[Violation]:
    col = df[rule.column].astype(str)
    pattern = re.compile(rule.pattern)
    violations = []
    for idx in df.index:
        val = col.iloc[idx]
        if val in ("nan", "None", ""):
            continue  # empty/null handled by NotNullRule
        if not pattern.search(val):
            violations.append(Violation(
                rule_type="regex",
                column=rule.column,
                row_index=idx,
                expected=f"matches {rule.pattern}",
                actual_value=val,
                rule=rule,
            ))
    return violations


def _check_enum(df: pd.DataFrame, rule: EnumRule) -> list[Violation]:
    col = df[rule.column]
    allowed = set(rule.values)
    # Case-insensitive comparison
    allowed_lower = {str(v).lower() for v in allowed}
    violations = []
    for idx in df.index:
        val = col.iloc[idx]
        if pd.isna(val):
            continue
        if str(val).lower() not in allowed_lower:
            violations.append(Violation(
                rule_type="enum",
                column=rule.column,
                row_index=idx,
                expected=f"one of {sorted(allowed)}",
                actual_value=val,
                rule=rule,
            ))
    return violations


def _check_dtype(df: pd.DataFrame, rule: DtypeRule) -> list[Violation]:
    col = df[rule.column]
    violations = []
    for idx in df.index:
        val = col.iloc[idx]
        if pd.isna(val):
            continue
        if not _is_compatible_dtype(val, rule.expected_dtype):
            violations.append(Violation(
                rule_type="dtype",
                column=rule.column,
                row_index=idx,
                expected=f"dtype {rule.expected_dtype}",
                actual_value=val,
                rule=rule,
            ))
    return violations


def _is_compatible_dtype(val: Any, expected: str) -> bool:
    if expected == "integer":
        try:
            f = float(val)
            return f == int(f)
        except (ValueError, TypeError):
            return False
    elif expected == "float":
        try:
            float(val)
            return True
        except (ValueError, TypeError):
            return False
    elif expected == "date":
        try:
            pd.Timestamp(val)
            return True
        except (ValueError, TypeError):
            return False
    elif expected == "string":
        return isinstance(val, str)
    return True


def _check_not_null(df: pd.DataFrame, rule: NotNullRule) -> list[Violation]:
    col = df[rule.column]
    violations = []
    for idx in df.index:
        if pd.isna(col.iloc[idx]):
            violations.append(Violation(
                rule_type="not_null",
                column=rule.column,
                row_index=idx,
                expected="not null",
                actual_value=None,
                rule=rule,
            ))
    return violations


def _check_unique(df: pd.DataFrame, rule: UniqueRule) -> list[Violation]:
    col = df[rule.column]
    duplicated_mask = col.duplicated(keep=False)
    violations = []
    seen_vals: set = set()
    for idx in df.index:
        val = col.iloc[idx]
        if pd.isna(val):
            continue
        if duplicated_mask.iloc[idx] and val in seen_vals:
            violations.append(Violation(
                rule_type="unique",
                column=rule.column,
                row_index=idx,
                expected="unique value",
                actual_value=val,
                rule=rule,
            ))
        seen_vals.add(val)
    return violations


def _check_cross_column(
    df: pd.DataFrame,
    rule: CrossColumnRule,
    maps: dict[str, dict],
) -> list[Violation]:
    if not all(c in df.columns for c in rule.columns):
        return []

    violations = []
    if rule.condition == "functional_dependency" and len(rule.columns) == 2:
        col_a, col_b = rule.columns
        map_key = f"{col_a}->{col_b}"
        mapping = maps.get(map_key, {})
        if not mapping:
            return []  # No mapping provided — can't validate
        for idx in df.index:
            val_a = df[col_a].iloc[idx]
            val_b = df[col_b].iloc[idx]
            if pd.isna(val_a) or pd.isna(val_b):
                continue
            expected_b = mapping.get(str(val_a))
            if expected_b is not None and str(val_b) != str(expected_b):
                violations.append(Violation(
                    rule_type="cross_column",
                    column=f"{col_a},{col_b}",
                    row_index=idx,
                    expected=f"{col_a}={val_a} -> {col_b}={expected_b}",
                    actual_value=val_b,
                    rule=rule,
                ))
    elif rule.condition == "ordering" and len(rule.columns) == 2:
        col_a, col_b = rule.columns
        for idx in df.index:
            val_a = pd.to_numeric(df[col_a].iloc[idx], errors="coerce")
            val_b = pd.to_numeric(df[col_b].iloc[idx], errors="coerce")
            if pd.isna(val_a) or pd.isna(val_b):
                continue
            if val_a > val_b:
                violations.append(Violation(
                    rule_type="cross_column",
                    column=f"{col_a},{col_b}",
                    row_index=idx,
                    expected=f"{col_a} <= {col_b}",
                    actual_value=f"{val_a} > {val_b}",
                    rule=rule,
                ))
    return violations
```

Update `server/rules/__init__.py` to export validator:

```python
# Add to server/rules/__init__.py
from .validator import validate, Violation, compute_semantic_score
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/shivanshu.s/deployments/proj/outlier_clean_data/.claude/worktrees/env-v2 && python -m pytest tests/test_rules.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add server/rules/validator.py server/rules/__init__.py tests/test_rules.py
git commit -m "feat(rules): validator checks DataFrame against 7 rule types"
```

---

## Task 3: Rule Inferrer — Domain Heuristics + Statistical

**Files:**
- Create: `server/rules/inferrer.py`
- Test: `tests/test_rules.py` (append)

- [ ] **Step 1: Write failing tests for inferrer**

Append to `tests/test_rules.py`:

```python
from server.rules.inferrer import infer_rules


class TestInferrerDomainHeuristics:
    def test_age_column_gets_range_rule(self):
        df = pd.DataFrame({"age": [25, 35, 45, 55]})
        rules = infer_rules(df)
        range_rules = [r for r in rules if r.rule_type == "range" and r.column == "age"]
        assert len(range_rules) == 1
        assert range_rules[0].min_val == 0
        assert range_rules[0].max_val == 120
        assert range_rules[0].source == "heuristic"

    def test_email_column_gets_regex_rule(self):
        df = pd.DataFrame({"email": ["a@b.com", "c@d.org"]})
        rules = infer_rules(df)
        regex_rules = [r for r in rules if r.rule_type == "regex" and r.column == "email"]
        assert len(regex_rules) == 1
        assert regex_rules[0].source == "heuristic"

    def test_id_column_gets_unique_and_not_null(self):
        df = pd.DataFrame({"passenger_id": [1, 2, 3]})
        rules = infer_rules(df)
        unique_rules = [r for r in rules if r.rule_type == "unique" and r.column == "passenger_id"]
        not_null_rules = [r for r in rules if r.rule_type == "not_null" and r.column == "passenger_id"]
        assert len(unique_rules) == 1
        assert len(not_null_rules) == 1

    def test_salary_column_gets_nonneg_range(self):
        df = pd.DataFrame({"salary": [50000, 80000, 120000]})
        rules = infer_rules(df)
        range_rules = [r for r in rules if r.rule_type == "range" and r.column == "salary"]
        assert len(range_rules) == 1
        assert range_rules[0].min_val == 0


class TestInferrerStatistical:
    def test_numeric_column_gets_expanded_range(self):
        df = pd.DataFrame({"score": [10.0, 20.0, 30.0, 40.0, 50.0]})
        rules = infer_rules(df)
        range_rules = [r for r in rules if r.rule_type == "range" and r.column == "score"]
        assert len(range_rules) == 1
        r = range_rules[0]
        # 20% expansion: min=10-8=2, max=50+8=58, clamped at 0 since all >= 0
        assert r.min_val == 0  # clamped
        assert r.max_val == pytest.approx(58.0)
        assert r.source == "statistical"

    def test_categorical_column_gets_enum(self):
        df = pd.DataFrame({"color": ["red", "blue", "green", "red", "blue"]})
        rules = infer_rules(df)
        enum_rules = [r for r in rules if r.rule_type == "enum" and r.column == "color"]
        assert len(enum_rules) == 1
        assert set(enum_rules[0].values) == {"red", "blue", "green"}

    def test_high_cardinality_column_no_enum(self):
        df = pd.DataFrame({"name": [f"person_{i}" for i in range(50)]})
        rules = infer_rules(df)
        enum_rules = [r for r in rules if r.rule_type == "enum" and r.column == "name"]
        assert len(enum_rules) == 0  # too many unique values

    def test_no_null_column_gets_not_null_rule(self):
        df = pd.DataFrame({"val": [1, 2, 3, 4, 5]})
        rules = infer_rules(df)
        nn_rules = [r for r in rules if r.rule_type == "not_null" and r.column == "val"]
        assert len(nn_rules) == 1

    def test_column_with_nulls_no_not_null_rule(self):
        df = pd.DataFrame({"val": [1, None, 3, None, 5]})
        rules = infer_rules(df)
        nn_rules = [r for r in rules if r.rule_type == "not_null" and r.column == "val"]
        assert len(nn_rules) == 0


class TestInferrerCrossColumn:
    def test_detects_functional_dependency(self):
        df = pd.DataFrame({
            "code": ["A", "B", "A", "C", "B", "A", "C"],
            "name": ["Alpha", "Beta", "Alpha", "Charlie", "Beta", "Alpha", "Charlie"],
        })
        rules = infer_rules(df)
        cross = [r for r in rules if r.rule_type == "cross_column"]
        assert len(cross) >= 1
        assert any("code" in r.columns and "name" in r.columns for r in cross)

    def test_no_false_dependency(self):
        df = pd.DataFrame({
            "x": [1, 2, 3, 4, 5],
            "y": [10, 20, 30, 40, 50],  # correlated but not functional dep
        })
        rules = infer_rules(df)
        cross = [r for r in rules if r.rule_type == "cross_column"]
        # Numeric columns shouldn't produce functional deps
        assert len(cross) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/shivanshu.s/deployments/proj/outlier_clean_data/.claude/worktrees/env-v2 && python -m pytest tests/test_rules.py::TestInferrerDomainHeuristics tests/test_rules.py::TestInferrerStatistical tests/test_rules.py::TestInferrerCrossColumn -v`
Expected: FAIL — `cannot import name 'infer_rules'`

- [ ] **Step 3: Implement inferrer**

```python
# server/rules/inferrer.py
"""Auto-infer semantic rules from clean DataFrames.

Three layers:
1. Domain heuristics — recognizes common column names
2. Statistical inference — data-driven rules for unmatched columns
3. Cross-column — functional dependency detection
"""
from __future__ import annotations

import re
from typing import Optional

import numpy as np
import pandas as pd

from .types import (
    CrossColumnRule,
    DtypeRule,
    EnumRule,
    NotNullRule,
    RangeRule,
    RegexRule,
    Rule,
    UniqueRule,
)

# ── Domain heuristic maps ────────────────────────────────────────────────

# Column name patterns -> known domain rules.
# Matched case-insensitively against the column name.
_RANGE_HEURISTICS: dict[tuple[str, ...], tuple[float, float]] = {
    ("age", "years", "year_of_age"): (0, 120),
    ("salary", "income", "wage", "wages", "pay", "earnings"): (0, float("inf")),
    ("price", "cost", "amount", "total", "fee", "charge"): (0, float("inf")),
    ("rating",): (0, 10),
    ("percentage", "pct", "percent"): (0, 100),
}

_REGEX_HEURISTICS: dict[tuple[str, ...], str] = {
    ("email", "email_address", "e_mail"): r"^[\w.+-]+@[\w-]+\.[\w.]+$",
    ("zip", "zipcode", "zip_code", "postal_code", "pincode", "pin_code"): r"^\d{4,6}$",
    ("phone", "phone_number", "telephone", "tel"): r"^[\d\s\-\+\(\)]{7,20}$",
    ("url", "website", "homepage", "link"): r"^https?://",
    ("ssn", "social_security"): r"^\d{3}-?\d{2}-?\d{4}$",
}

_ID_PATTERNS: tuple[str, ...] = (
    "_id", "id_", "identifier", "key", "code",
)


def _matches_heuristic(col_name: str, patterns: tuple[str, ...]) -> bool:
    """Check if a column name matches any pattern (case-insensitive, exact or suffix)."""
    lower = col_name.lower().strip()
    for pat in patterns:
        if lower == pat or lower.endswith(f"_{pat}") or lower.startswith(f"{pat}_"):
            return True
    return False


def _is_id_column(col_name: str) -> bool:
    lower = col_name.lower().strip()
    if lower == "id":
        return True
    for pat in _ID_PATTERNS:
        if pat in lower:
            return True
    return False


# ── Main entry point ─────────────────────────────────────────────────────

_MAX_ENUM_CARDINALITY = 30


def infer_rules(
    df: pd.DataFrame,
    domain: Optional[str] = None,
) -> list[Rule]:
    """Infer semantic rules from a clean DataFrame.

    Args:
        df: Clean (uncorrupted) DataFrame.
        domain: Optional dataset domain hint (e.g., "census", "health").

    Returns:
        List of inferred Rule objects.
    """
    rules: list[Rule] = []
    heuristic_cols: set[str] = set()  # columns handled by heuristics

    # Layer 1: Domain heuristics
    for col in df.columns:
        col_rules = _infer_heuristic(col, df)
        if col_rules:
            rules.extend(col_rules)
            heuristic_cols.add(col)

    # Layer 2: Statistical inference (for columns not matched by heuristics)
    for col in df.columns:
        rules.extend(_infer_statistical(col, df, skip_range=col in heuristic_cols))

    # Layer 3: Cross-column dependencies
    rules.extend(_infer_cross_column(df))

    return rules


def _infer_heuristic(col: str, df: pd.DataFrame) -> list[Rule]:
    """Apply domain-heuristic rules for known column name patterns."""
    rules: list[Rule] = []

    # Range heuristics
    for patterns, (lo, hi) in _RANGE_HEURISTICS.items():
        if _matches_heuristic(col, patterns):
            rules.append(RangeRule(column=col, min_val=lo, max_val=hi, source="heuristic"))
            break

    # Regex heuristics
    for patterns, pattern in _REGEX_HEURISTICS.items():
        if _matches_heuristic(col, patterns):
            rules.append(RegexRule(column=col, pattern=pattern, source="heuristic"))
            break

    # ID column heuristics
    if _is_id_column(col):
        rules.append(UniqueRule(column=col, source="heuristic"))
        rules.append(NotNullRule(column=col, source="heuristic"))

    return rules


def _infer_statistical(col: str, df: pd.DataFrame, skip_range: bool = False) -> list[Rule]:
    """Data-driven rule inference for a single column."""
    rules: list[Rule] = []
    series = df[col]

    # Not-null: if 0% nulls in clean data
    if series.notna().all() and not _is_id_column(col):
        # Don't duplicate not_null if ID heuristic already added it
        rules.append(NotNullRule(column=col, source="statistical"))

    # Numeric range (20% expansion)
    numeric = pd.to_numeric(series.dropna(), errors="coerce").dropna()
    if not skip_range and len(numeric) > 0 and len(numeric) / max(len(series.dropna()), 1) > 0.8:
        lo, hi = float(numeric.min()), float(numeric.max())
        spread = hi - lo if hi != lo else abs(hi) * 0.2 or 1.0
        expanded_lo = lo - 0.2 * spread
        expanded_hi = hi + 0.2 * spread
        # Clamp to 0 if all observed values are non-negative
        if lo >= 0:
            expanded_lo = 0.0
        rules.append(RangeRule(column=col, min_val=expanded_lo, max_val=expanded_hi, source="statistical"))

    # Categorical enum (low cardinality string columns)
    if series.dtype == object or str(series.dtype) == "category":
        non_null = series.dropna()
        unique_vals = non_null.unique()
        if 1 < len(unique_vals) <= _MAX_ENUM_CARDINALITY:
            rules.append(EnumRule(column=col, values=sorted(str(v) for v in unique_vals), source="statistical"))

    return rules


def _infer_cross_column(df: pd.DataFrame) -> list[Rule]:
    """Detect functional dependencies between categorical column pairs."""
    rules: list[Rule] = []
    cat_cols = [
        c for c in df.columns
        if (df[c].dtype == object or str(df[c].dtype) == "category")
        and df[c].nunique() <= _MAX_ENUM_CARDINALITY
    ]

    for i, col_a in enumerate(cat_cols):
        for col_b in cat_cols[i + 1:]:
            if _is_functional_dependency(df, col_a, col_b):
                mapping_str = _describe_mapping(df, col_a, col_b)
                rules.append(CrossColumnRule(
                    columns=[col_a, col_b],
                    condition="functional_dependency",
                    hint=f"Each {col_a} value maps to exactly one {col_b} value: {mapping_str}",
                    source="cross_column_inference",
                ))
            elif _is_functional_dependency(df, col_b, col_a):
                mapping_str = _describe_mapping(df, col_b, col_a)
                rules.append(CrossColumnRule(
                    columns=[col_b, col_a],
                    condition="functional_dependency",
                    hint=f"Each {col_b} value maps to exactly one {col_a} value: {mapping_str}",
                    source="cross_column_inference",
                ))

    return rules


def _is_functional_dependency(df: pd.DataFrame, col_a: str, col_b: str) -> bool:
    """Return True if col_a -> col_b is a functional dependency (each A value maps to exactly one B)."""
    pairs = df[[col_a, col_b]].dropna()
    if len(pairs) < 3:
        return False
    grouped = pairs.groupby(col_a)[col_b].nunique()
    return bool((grouped == 1).all())


def _describe_mapping(df: pd.DataFrame, col_a: str, col_b: str, max_entries: int = 8) -> str:
    """Produce a short description of the A->B mapping."""
    pairs = df[[col_a, col_b]].dropna().drop_duplicates()
    mapping = dict(zip(pairs[col_a].astype(str), pairs[col_b].astype(str)))
    items = list(mapping.items())[:max_entries]
    parts = [f"{a}={b}" for a, b in items]
    suffix = f" (+{len(mapping) - max_entries} more)" if len(mapping) > max_entries else ""
    return ", ".join(parts) + suffix
```

Update `server/rules/__init__.py`:

```python
# Add to server/rules/__init__.py
from .inferrer import infer_rules
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/shivanshu.s/deployments/proj/outlier_clean_data/.claude/worktrees/env-v2 && python -m pytest tests/test_rules.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add server/rules/inferrer.py server/rules/__init__.py tests/test_rules.py
git commit -m "feat(rules): auto-infer rules from clean data (heuristic + statistical + cross-column)"
```

---

## Task 4: Catalog Enricher

**Files:**
- Create: `server/rules/catalog_enricher.py`
- Test: `tests/test_rules.py` (append)

- [ ] **Step 1: Write failing test for enricher**

Append to `tests/test_rules.py`:

```python
import json
import tempfile
import os
from server.rules.catalog_enricher import enrich_catalog


class TestCatalogEnricher:
    def test_enriches_single_dataset(self, tmp_path):
        # Create a mini catalog
        catalog = [
            {"id": "test_ds", "filename": "test.csv", "domain": "test"}
        ]
        catalog_path = tmp_path / "catalog.json"
        catalog_path.write_text(json.dumps(catalog))

        # Create a clean CSV
        clean_dir = tmp_path / "clean"
        clean_dir.mkdir()
        df = pd.DataFrame({"age": [25, 35, 45], "name": ["Alice", "Bob", "Charlie"]})
        df.to_csv(clean_dir / "test.csv", index=False)

        # Enrich
        enrich_catalog(str(catalog_path), str(clean_dir))

        # Verify
        enriched = json.loads(catalog_path.read_text())
        assert "rules" in enriched[0]
        rules = enriched[0]["rules"]
        assert any(r["type"] == "range" and r["column"] == "age" for r in rules)

    def test_manual_overrides_win(self, tmp_path):
        catalog = [
            {
                "id": "test_ds",
                "filename": "test.csv",
                "domain": "test",
                "rules_override": [
                    {"type": "range", "column": "age", "min": 18, "max": 65, "source": "manual"}
                ],
            }
        ]
        catalog_path = tmp_path / "catalog.json"
        catalog_path.write_text(json.dumps(catalog))

        clean_dir = tmp_path / "clean"
        clean_dir.mkdir()
        df = pd.DataFrame({"age": [25, 35, 45]})
        df.to_csv(clean_dir / "test.csv", index=False)

        enrich_catalog(str(catalog_path), str(clean_dir))

        enriched = json.loads(catalog_path.read_text())
        range_rules = [r for r in enriched[0]["rules"] if r["type"] == "range" and r["column"] == "age"]
        assert len(range_rules) == 1
        assert range_rules[0]["min"] == 18  # manual override
        assert range_rules[0]["source"] == "manual"

    def test_skips_missing_csv(self, tmp_path):
        catalog = [{"id": "missing", "filename": "nope.csv"}]
        catalog_path = tmp_path / "catalog.json"
        catalog_path.write_text(json.dumps(catalog))
        clean_dir = tmp_path / "clean"
        clean_dir.mkdir()

        enrich_catalog(str(catalog_path), str(clean_dir))

        enriched = json.loads(catalog_path.read_text())
        assert "rules" not in enriched[0]  # no clean file, no rules
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/shivanshu.s/deployments/proj/outlier_clean_data/.claude/worktrees/env-v2 && python -m pytest tests/test_rules.py::TestCatalogEnricher -v`
Expected: FAIL

- [ ] **Step 3: Implement catalog enricher**

```python
# server/rules/catalog_enricher.py
"""Enrich catalog.json with auto-inferred semantic rules.

Usage:
    python -m server.rules.catalog_enricher datasets/catalog.json datasets/clean/
"""
from __future__ import annotations

import json
import logging
import os
import sys

import pandas as pd

from .inferrer import infer_rules
from .types import rule_from_dict, rule_to_dict

logger = logging.getLogger(__name__)


def enrich_catalog(catalog_path: str, clean_dir: str) -> None:
    """Read catalog, infer rules for each dataset, merge overrides, write back.

    Args:
        catalog_path: Path to catalog.json.
        clean_dir: Directory containing clean CSV files.
    """
    with open(catalog_path) as f:
        catalog = json.load(f)

    for entry in catalog:
        filename = entry.get("filename", "")
        csv_path = os.path.join(clean_dir, filename)
        if not os.path.exists(csv_path):
            logger.warning("Skipping %s: no clean file at %s", entry.get("id", "?"), csv_path)
            continue

        try:
            csv_params = entry.get("csv_params", {})
            df = pd.read_csv(csv_path, **csv_params, nrows=entry.get("max_rows"))
        except Exception as exc:
            logger.warning("Skipping %s: failed to read CSV: %s", entry.get("id", "?"), exc)
            continue

        domain = entry.get("domain")
        inferred = infer_rules(df, domain=domain)
        inferred_dicts = [rule_to_dict(r) for r in inferred]

        # Merge with manual overrides
        overrides = entry.get("rules_override", [])
        merged = _merge_rules(inferred_dicts, overrides)

        entry["rules"] = merged

    with open(catalog_path, "w") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)

    logger.info("Enriched %d datasets in %s", len(catalog), catalog_path)


def _merge_rules(inferred: list[dict], overrides: list[dict]) -> list[dict]:
    """Merge inferred rules with manual overrides. Manual wins for same column+type."""
    # Build a key for dedup: (type, column) or (type, columns_tuple)
    def _key(r: dict) -> tuple:
        if "column" in r:
            return (r["type"], r["column"])
        elif "columns" in r:
            return (r["type"], tuple(r["columns"]))
        return (r["type"], "")

    override_keys = {_key(r) for r in overrides}
    # Keep inferred rules that aren't overridden
    result = [r for r in inferred if _key(r) not in override_keys]
    result.extend(overrides)
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) < 3:
        print(f"Usage: python -m server.rules.catalog_enricher <catalog.json> <clean_dir>")
        sys.exit(1)
    enrich_catalog(sys.argv[1], sys.argv[2])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/shivanshu.s/deployments/proj/outlier_clean_data/.claude/worktrees/env-v2 && python -m pytest tests/test_rules.py::TestCatalogEnricher -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add server/rules/catalog_enricher.py tests/test_rules.py
git commit -m "feat(rules): catalog enricher auto-populates rules in catalog.json"
```

---

## Task 5: Grader — Semantic Score (5th Dimension)

**Files:**
- Create: `tests/test_grader_semantic.py`
- Modify: `server/grader.py:588-662`

- [ ] **Step 1: Write failing tests for semantic grading**

```python
# tests/test_grader_semantic.py
import pytest
import pandas as pd
from server.grader import grade
from server.rules.types import RangeRule, EnumRule, NotNullRule


@pytest.fixture
def clean_df():
    return pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "age": [25, 35, 45, 55, 65],
        "sex": ["male", "female", "male", "female", "male"],
        "score": [88.5, 92.0, 75.0, 81.0, 90.0],
    })


@pytest.fixture
def error_map():
    return {
        "cell_errors": {
            "0,age": {"severity": 1.0, "clean_value": 25, "corruption": "business_rule_violation"},
        },
        "spurious_rows": {},
        "missing_rows": {},
    }


@pytest.fixture
def rules():
    return [
        RangeRule(column="age", min_val=0, max_val=120, source="heuristic"),
        EnumRule(column="sex", values=["male", "female"], source="statistical"),
        NotNullRule(column="score", source="statistical"),
    ]


class TestSemanticScoreInGrade:
    def test_perfect_result_with_rules(self, clean_df, error_map, rules):
        """Perfect result should get semantic_score near 1.0."""
        _, reward = grade(
            clean_df, clean_df, error_map,
            transform_steps=3, min_transform_steps=2, max_transform_steps=10,
            rules=rules,
        )
        assert reward > 0.9

    def test_rule_violations_reduce_score(self, clean_df, error_map, rules):
        """Result with semantic violations should score lower than without."""
        bad_result = clean_df.copy()
        bad_result.loc[0, "age"] = -5      # range violation
        bad_result.loc[1, "sex"] = "alien"  # enum violation

        _, reward_bad = grade(
            clean_df, bad_result, error_map,
            transform_steps=3, min_transform_steps=2, max_transform_steps=10,
            rules=rules,
        )
        _, reward_good = grade(
            clean_df, clean_df, error_map,
            transform_steps=3, min_transform_steps=2, max_transform_steps=10,
            rules=rules,
        )
        assert reward_good > reward_bad

    def test_no_rules_defaults_to_full_score(self, clean_df, error_map):
        """Without rules, semantic dimension should be 1.0 (no penalty)."""
        _, reward_no_rules = grade(
            clean_df, clean_df, error_map,
            transform_steps=3, min_transform_steps=2, max_transform_steps=10,
        )
        _, reward_with_rules = grade(
            clean_df, clean_df, error_map,
            transform_steps=3, min_transform_steps=2, max_transform_steps=10,
            rules=[],
        )
        assert reward_no_rules == pytest.approx(reward_with_rules)

    def test_reward_still_clamped_0_to_1(self, clean_df, error_map, rules):
        """Reward must stay in [0.0, 1.0] regardless of violations."""
        terrible = clean_df.copy()
        terrible["age"] = [-999, -999, -999, -999, -999]
        terrible["sex"] = ["x", "x", "x", "x", "x"]
        terrible["score"] = [None, None, None, None, None]

        _, reward = grade(
            clean_df, terrible, error_map,
            transform_steps=3, min_transform_steps=2, max_transform_steps=10,
            rules=rules,
        )
        assert 0.0 <= reward <= 1.0

    def test_weight_rebalance(self, clean_df, error_map, rules):
        """With rules, the semantic dimension should have 10% weight."""
        # We verify indirectly: a result that's perfect on cells but violates rules
        # should score less than a result perfect on everything
        rule_violating = clean_df.copy()
        rule_violating.loc[0, "age"] = 200  # range violation

        _, reward = grade(
            clean_df, rule_violating, error_map,
            transform_steps=3, min_transform_steps=2, max_transform_steps=10,
            rules=rules,
        )
        # Semantic is 10%, so a violation should reduce score by roughly that amount
        # (not exactly, since cell score also affected)
        assert reward < 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/shivanshu.s/deployments/proj/outlier_clean_data/.claude/worktrees/env-v2 && python -m pytest tests/test_grader_semantic.py -v`
Expected: FAIL — `grade() got an unexpected keyword argument 'rules'`

- [ ] **Step 3: Modify grader to add semantic dimension**

In `server/grader.py`, make these changes:

**Add import at top of file** (after existing imports):
```python
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from server.rules.types import Rule
```

At runtime, import conditionally inside the function to avoid circular imports.

**Modify the `grade()` function** (replace lines 588-662):

```python
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
) -> tuple[dict[str, str], float]:
    """Grade the agent's result using multi-level scoring.

    Five scoring dimensions:
        - Schema (0.15): column name matching + type compatibility
        - Row (0.15): row count correctness, spurious/missing row handling
        - Cell (0.50): severity-weighted cell error resolution
        - Distribution (0.10): imputation quality for null-filled columns
        - Semantic (0.10): rule compliance (defaults to 1.0 if no rules)

    Args:
        clean_df: Ground truth DataFrame.
        result_df: Agent's cleaned DataFrame.
        error_map: Dict with keys "cell_errors", "spurious_rows", "missing_rows".
        transform_steps: Number of transform steps the agent used.
        min_transform_steps: Minimum expected steps (efficiency baseline).
        max_transform_steps: Maximum allowed steps.
        explore_steps: Total number of explore actions taken.
        explore_timeouts: Number of explore actions that timed out or failed.
        undo_count: Number of undo actions taken.
        validate_count: Number of validate actions taken.
        explore_cost_per_step: Penalty per successful explore action.
        explore_timeout_cost: Penalty per timed-out/failed explore action.
        undo_cost: Penalty per undo action.
        validate_cost: Penalty per validate action.
        rules: Optional list of Rule objects for semantic validation.
        cross_column_maps: Optional cross-column mapping dicts for validation.

    Returns:
        (error_status, reward) where error_status maps each error key to one of
        "fixed", "wrong_value", or "unfixed", and reward is in [0.0, 1.0].
    """
    s_score = schema_score(clean_df, result_df)
    row_map = match_rows_by_content(clean_df, result_df)
    r_score = row_score(clean_df, result_df, error_map)

    # Identify imputed columns (those with null errors)
    imputed_cols: set[str] = set()
    for key, info in error_map.get("cell_errors", {}).items():
        if info.get("corruption") in ("inject_nulls", "null_injected"):
            _, col = _parse_error_key(key)
            imputed_cols.add(col)

    c_score, error_status = _cell_score_full(clean_df, result_df, error_map, row_map)
    d_score = distribution_score(clean_df, result_df, imputed_cols)

    # Semantic score (5th dimension)
    sem_score = _compute_semantic(result_df, rules, cross_column_maps)

    # Weights: with rules present, use 5-dimension weights; otherwise legacy 4-dimension
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
    return error_status, reward


def _compute_semantic(
    result_df: pd.DataFrame,
    rules: list | None,
    cross_column_maps: dict | None,
) -> float:
    """Compute semantic score. Returns 1.0 if no rules."""
    if not rules:
        return 1.0
    from server.rules.validator import compute_semantic_score
    return compute_semantic_score(result_df, rules, cross_column_maps)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/shivanshu.s/deployments/proj/outlier_clean_data/.claude/worktrees/env-v2 && python -m pytest tests/test_grader_semantic.py tests/test_grader.py -v`
Expected: All tests PASS (both new semantic tests AND existing grader tests)

- [ ] **Step 5: Commit**

```bash
git add server/grader.py tests/test_grader_semantic.py
git commit -m "feat(grader): add semantic_score as 5th grading dimension (10% weight)"
```

---

## Task 6: Corruption Integration — Rule-Aware Business Rule Violation

**Files:**
- Modify: `server/corruption/value_corruptions.py:842-890`
- Modify: `server/corruption/pipeline.py:42-164`
- Test: `tests/test_corruption_pipeline.py` (append)

- [ ] **Step 1: Write failing tests**

Append to `tests/test_corruption_pipeline.py`:

```python
from server.rules.types import RangeRule, EnumRule, NotNullRule, RegexRule, UniqueRule, rule_to_dict


class TestRuleAwareCorruption:
    def test_pipeline_accepts_rules(self, sample_df):
        rules = [
            RangeRule(column="age", min_val=0, max_val=120, source="heuristic"),
            EnumRule(column="city", values=["NYC", "LA", "CHI"], source="statistical"),
        ]
        pipe = CorruptionPipeline(seed=42, difficulty="medium")
        pipe.select_format()
        dirty_df, error_map, sev, meta = pipe.corrupt(sample_df, rules=rules)
        assert isinstance(dirty_df, pd.DataFrame)

    def test_rule_violations_tracked_in_error_map(self, sample_df):
        rules = [
            RangeRule(column="age", min_val=0, max_val=120, source="heuristic"),
        ]
        pipe = CorruptionPipeline(seed=99, difficulty="easy")
        pipe.select_format()
        dirty_df, error_map, sev, meta = pipe.corrupt(sample_df, rules=rules)
        # If business_rule_violation was selected, check for rule_id in errors
        biz_errors = [
            v for v in error_map.get("cell_errors", {}).values()
            if v.get("corruption") == "business_rule_violation"
        ]
        # May or may not have biz errors depending on random selection,
        # but the pipeline should not crash
        assert isinstance(error_map, dict)

    def test_pipeline_without_rules_still_works(self, sample_df):
        pipe = CorruptionPipeline(seed=42, difficulty="hard")
        pipe.select_format()
        dirty_df, error_map, sev, meta = pipe.corrupt(sample_df)
        assert isinstance(dirty_df, pd.DataFrame)
        assert "cell_errors" in error_map
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/shivanshu.s/deployments/proj/outlier_clean_data/.claude/worktrees/env-v2 && python -m pytest tests/test_corruption_pipeline.py::TestRuleAwareCorruption -v`
Expected: FAIL — `corrupt() got an unexpected keyword argument 'rules'`

- [ ] **Step 3: Modify pipeline.corrupt() to accept rules**

In `server/corruption/pipeline.py`, modify the `corrupt` method signature to accept `rules`:

```python
def corrupt(
    self, clean_df: pd.DataFrame, rules: list | None = None,
) -> tuple[pd.DataFrame, dict, dict, dict]:
```

Store rules on self so `business_rule_violation` can access them:

```python
self._rules = rules or []
```

Pass `rules=self._rules` through `**kwargs` when calling `business_rule_violation`:

In the corruption function call loop, add:
```python
kwargs = {}
if name == "business_rule_violation" and self._rules:
    kwargs["rules"] = self._rules
result_df = fn(result_df, cols, frac, error_log, clean_df, self.rng, self.py_rng, **kwargs)
```

- [ ] **Step 4: Modify business_rule_violation to use rules**

In `server/corruption/value_corruptions.py`, update `business_rule_violation` to accept and use rules from kwargs:

```python
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

    If rules are provided via kwargs["rules"], uses them to create targeted
    violations. Otherwise falls back to generic impossible-value injection.
    """
    if error_log is None:
        error_log = []
    if rng is None:
        rng = np.random.default_rng()
    result = df.copy()
    rules = kwargs.get("rules", [])
    n_rows = len(result)
    n_corrupt = max(1, int(n_rows * fraction))
    rows = rng.choice(n_rows, size=min(n_corrupt, n_rows), replace=False)

    if rules:
        # Use rules to create targeted violations
        from server.rules.types import RangeRule, RegexRule, EnumRule, NotNullRule, UniqueRule
        for idx in rows:
            # Pick a random rule that applies to available columns
            applicable = [
                r for r in rules
                if hasattr(r, "column") and r.column in result.columns
            ]
            if not applicable:
                break
            rule = py_rng.choice(applicable)
            col = rule.column

            if isinstance(rule, RangeRule):
                # Inject out-of-range value
                if rng.random() < 0.5:
                    bad_val = rule.min_val - rng.uniform(1, abs(rule.max_val - rule.min_val) * 0.5 + 1)
                else:
                    if rule.max_val == float("inf"):
                        bad_val = -rng.uniform(1, 1000)
                    else:
                        bad_val = rule.max_val + rng.uniform(1, abs(rule.max_val - rule.min_val) * 0.5 + 1)
                clean_val = result.at[idx, col]
                result.at[idx, col] = bad_val
            elif isinstance(rule, EnumRule):
                # Inject invalid category
                bad_val = f"INVALID_{rng.integers(100, 999)}"
                clean_val = result.at[idx, col]
                result.at[idx, col] = bad_val
            elif isinstance(rule, NotNullRule):
                clean_val = result.at[idx, col]
                result.at[idx, col] = None
            elif isinstance(rule, RegexRule):
                clean_val = result.at[idx, col]
                result.at[idx, col] = f"!!!INVALID_{rng.integers(100)}"
            elif isinstance(rule, UniqueRule):
                # Duplicate an existing value
                other_idx = rng.choice(n_rows)
                clean_val = result.at[idx, col]
                result.at[idx, col] = result.at[other_idx, col]
            else:
                continue

            error_log.append({
                "key": f"{idx},{col}",
                "severity": CORRUPTION_SEVERITY.get("business_rule_violation", 3.0),
                "clean_value": _safe_clean_value(clean_val),
                "corruption": "business_rule_violation",
                "rule_type": rule.rule_type,
            })
    else:
        # Fallback: generic impossible-value injection (existing behavior)
        numeric_cols = [c for c in columns if pd.api.types.is_numeric_dtype(result[c])]
        if not numeric_cols:
            return result
        for idx in rows:
            col = py_rng.choice(numeric_cols)
            val = result.at[idx, col]
            if pd.isna(val):
                continue
            clean_val = val
            choice = rng.integers(3)
            if choice == 0:
                result.at[idx, col] = -abs(float(val)) - rng.uniform(1, 100)
            elif choice == 1:
                result.at[idx, col] = float(val) * rng.uniform(10, 1000)
            else:
                result.at[idx, col] = -float(val)
            error_log.append({
                "key": f"{idx},{col}",
                "severity": CORRUPTION_SEVERITY.get("business_rule_violation", 3.0),
                "clean_value": _safe_clean_value(clean_val),
                "corruption": "business_rule_violation",
            })

    return result
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /Users/shivanshu.s/deployments/proj/outlier_clean_data/.claude/worktrees/env-v2 && python -m pytest tests/test_corruption_pipeline.py -v`
Expected: All tests PASS (both new and existing)

- [ ] **Step 6: Commit**

```bash
git add server/corruption/pipeline.py server/corruption/value_corruptions.py tests/test_corruption_pipeline.py
git commit -m "feat(corruption): rule-aware business_rule_violation uses semantic rules"
```

---

## Task 7: Environment Integration — Rules in Reset + Observation

**Files:**
- Modify: `server/environment.py:230-327`
- Modify: `models.py:151-166`

- [ ] **Step 1: Add `semantic_rules` field to observation model**

In `models.py`, add to `DataCleaningObservation`:

```python
class DataCleaningObservation(BaseObservation):
    task_id: str = ""
    task_description: str = ""
    constraints: List[str] = []
    data_summary: str = ""
    explore_result: Optional[str] = None
    transform_result: Optional[str] = None
    constraint_status: Dict[str, bool] = {}
    file_format: Optional[str] = None
    target_schema: Optional[Dict[str, str]] = None
    file_preview: Optional[str] = None
    diagnosis: Optional[str] = None
    validate_result: Optional[str] = None
    step_info: Optional[StepInfo] = None
    semantic_rules: List[Dict] = []  # NEW: rules for this dataset
```

- [ ] **Step 2: Modify environment.reset() to load and pass rules**

In `server/environment.py`, add to `reset()` after loading the dataset entry (around line 256):

```python
# Load semantic rules for this dataset
raw_rules = dataset_entry.get("rules", [])
self._rules = []
self._rules_dicts = raw_rules  # Keep dicts for observation
if raw_rules:
    try:
        from server.rules.types import rule_from_dict
        self._rules = [rule_from_dict(r) for r in raw_rules]
    except Exception:
        from rules.types import rule_from_dict
        self._rules = [rule_from_dict(r) for r in raw_rules]
```

Update the pipeline.corrupt() call to pass rules:

```python
dirty_df, error_map, _severity_map, pipeline_metadata = pipeline.corrupt(
    self._clean_df, difficulty, np_rng, py_rng, rules=self._rules,
)
```

Note: The current `pipeline.corrupt()` call at line 284 passes extra args (`difficulty, np_rng, py_rng`) that don't match the pipeline signature. This was already a discrepancy in the codebase — verify and align the call with the actual pipeline API when implementing.

- [ ] **Step 3: Pass rules into grade()**

In `server/environment.py`, modify `_regrade()` to pass rules:

```python
def _regrade(self) -> None:
    profile = self._profile
    self._error_status, self._current_reward = grade(
        self._clean_df,
        self._current_df,
        self._error_map,
        self._transform_steps,
        profile.get("min_transform_steps", 2),
        profile.get("max_transform_steps", 10),
        explore_steps=self._explore_steps_total,
        explore_timeouts=self._explore_timeouts,
        explore_cost_per_step=profile.get("explore_cost_per_step", _EXPLORE_COST_PER_STEP_DEFAULT),
        explore_timeout_cost=profile.get("explore_timeout_cost", _EXPLORE_TIMEOUT_COST_DEFAULT),
        undo_count=self._undo_count,
        validate_count=self._validate_uses,
        undo_cost=profile.get("undo_cost", _UNDO_COST_DEFAULT),
        validate_cost=profile.get("validate_cost", _VALIDATE_COST_DEFAULT),
        rules=self._rules,
    )
    self._error_summary_cache = summarize_errors(self._error_status, self._error_map)
```

- [ ] **Step 4: Include rules in observation**

In `_make_observation()`, add `semantic_rules=self._rules_dicts`:

```python
return DataCleaningObservation(
    # ... existing fields ...
    semantic_rules=self._rules_dicts,
)
```

- [ ] **Step 5: Run all tests**

Run: `cd /Users/shivanshu.s/deployments/proj/outlier_clean_data/.claude/worktrees/env-v2 && python -m pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add server/environment.py models.py
git commit -m "feat(env): load semantic rules at reset, pass to grader and observation"
```

---

## Task 8: Category Definitions

**Files:**
- Create: `server/corruption/categories.py`
- Create: `tests/test_categories.py`

- [ ] **Step 1: Write failing tests for categories**

```python
# tests/test_categories.py
import pytest
from server.corruption.categories import (
    CATEGORY_CORRUPTION_MAP,
    CATEGORY_FORMAT_MAP,
    CATEGORIES,
    get_corruptions_for_category,
    get_formats_for_category,
)
from server.corruption.value_corruptions import CORRUPTION_REGISTRY


class TestCategoryDefinitions:
    def test_all_six_categories_defined(self):
        assert set(CATEGORIES) == {"FP", "VR", "MD", "SR", "SV", "CP"}

    def test_all_corruption_map_entries_are_valid(self):
        registry_names = {entry["name"] for entry in CORRUPTION_REGISTRY}
        for cat, corruptions in CATEGORY_CORRUPTION_MAP.items():
            if corruptions is None:
                continue  # CP uses all
            for c in corruptions:
                assert c in registry_names, f"{c} in {cat} not in CORRUPTION_REGISTRY"

    def test_format_map_entries_are_valid(self):
        valid_formats = {"csv", "tsv", "json", "jsonl", "excel", "xml",
                         "fixed_width", "html_table", "sql_dump", "yaml"}
        for cat, formats in CATEGORY_FORMAT_MAP.items():
            for f in formats:
                assert f in valid_formats, f"{f} in {cat} not a valid format"

    def test_get_corruptions_for_known_category(self):
        corruptions = get_corruptions_for_category("VR")
        assert "type_mangle" in corruptions
        assert "decimal_shift" in corruptions
        assert "inject_nulls" not in corruptions  # MD, not VR

    def test_get_corruptions_for_compound(self):
        corruptions = get_corruptions_for_category("CP")
        # CP returns None = all corruptions allowed
        assert corruptions is None

    def test_get_formats_for_category(self):
        formats = get_formats_for_category("FP")
        assert "csv" not in formats  # FP uses non-CSV formats
        assert "json" in formats

    def test_get_formats_for_vr_is_csv(self):
        formats = get_formats_for_category("VR")
        assert formats == ["csv"]

    def test_invalid_category_raises(self):
        with pytest.raises(ValueError):
            get_corruptions_for_category("INVALID")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/shivanshu.s/deployments/proj/outlier_clean_data/.claude/worktrees/env-v2 && python -m pytest tests/test_categories.py -v`
Expected: FAIL

- [ ] **Step 3: Implement categories module**

```python
# server/corruption/categories.py
"""Benchmark category definitions.

Maps each of 6 skill categories to the corruption types and file formats
that stress-test that category. Used by CorruptionPipeline when
a category is specified to constrain what gets generated.
"""
from __future__ import annotations

CATEGORIES = ("FP", "VR", "MD", "SR", "SV", "CP")

# Corruption types per category. CP = None (any 7+ types from all categories).
CATEGORY_CORRUPTION_MAP: dict[str, list[str] | None] = {
    "FP": ["encoding_noise", "header_in_data"],  # + format-level corruptions
    "VR": [
        "type_mangle", "decimal_shift", "value_swap", "typo_injection",
        "unicode_homoglyph", "html_entity_leak", "leading_zero_strip",
    ],
    "MD": ["inject_nulls", "drop_rows"],
    "SR": ["duplicate_rows", "column_shift", "schema_drift", "header_in_data"],
    "SV": ["business_rule_violation", "unit_inconsistency", "outlier_injection"],
    "CP": None,
}

# File format pool per category.
CATEGORY_FORMAT_MAP: dict[str, list[str]] = {
    "FP": ["json", "jsonl", "excel", "xml", "html_table", "fixed_width", "sql_dump", "yaml"],
    "VR": ["csv"],
    "MD": ["csv"],
    "SR": ["csv", "tsv"],
    "SV": ["csv"],
    "CP": ["json", "excel", "xml", "html_table"],
}


def get_corruptions_for_category(category: str) -> list[str] | None:
    """Return the corruption type names for a category, or None for CP (all)."""
    if category not in CATEGORY_CORRUPTION_MAP:
        raise ValueError(f"Unknown category: {category!r}. Must be one of {CATEGORIES}")
    return CATEGORY_CORRUPTION_MAP[category]


def get_formats_for_category(category: str) -> list[str]:
    """Return the format pool for a category."""
    if category not in CATEGORY_FORMAT_MAP:
        raise ValueError(f"Unknown category: {category!r}. Must be one of {CATEGORIES}")
    return CATEGORY_FORMAT_MAP[category]
```

- [ ] **Step 4: Update `server/corruption/__init__.py` exports**

```python
from .categories import (
    CATEGORIES,
    CATEGORY_CORRUPTION_MAP,
    CATEGORY_FORMAT_MAP,
    get_corruptions_for_category,
    get_formats_for_category,
)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /Users/shivanshu.s/deployments/proj/outlier_clean_data/.claude/worktrees/env-v2 && python -m pytest tests/test_categories.py -v`
Expected: All 8 tests PASS

- [ ] **Step 6: Commit**

```bash
git add server/corruption/categories.py server/corruption/__init__.py tests/test_categories.py
git commit -m "feat(categories): define 6 benchmark categories with corruption/format maps"
```

---

## Task 9: Category-Aware Pipeline Generation

**Files:**
- Modify: `server/corruption/pipeline.py`
- Test: `tests/test_categories.py` (append)

- [ ] **Step 1: Write failing tests**

Append to `tests/test_categories.py`:

```python
import pandas as pd
import numpy as np
from server.corruption.pipeline import CorruptionPipeline


@pytest.fixture
def sample_df():
    rng = np.random.default_rng(42)
    n = 100
    return pd.DataFrame({
        "id": range(n),
        "age": rng.integers(18, 80, size=n),
        "name": [f"Person_{i}" for i in range(n)],
        "score": rng.uniform(0, 100, size=n).round(2),
        "city": rng.choice(["NYC", "LA", "CHI", "HOU", "PHX"], size=n),
        "active": rng.choice([True, False], size=n),
        "email": [f"person{i}@example.com" for i in range(n)],
    })


class TestCategoryAwarePipeline:
    def test_vr_category_only_uses_vr_corruptions(self, sample_df):
        pipe = CorruptionPipeline(seed=42, difficulty="medium", category="VR")
        pipe.select_format()
        dirty_df, error_map, sev, meta = pipe.corrupt(sample_df)
        applied = meta.get("corruptions_applied", [])
        vr_set = set(CATEGORY_CORRUPTION_MAP["VR"])
        for name in applied:
            assert name in vr_set, f"{name} not in VR category"

    def test_fp_category_forces_non_csv_format(self, sample_df):
        pipe = CorruptionPipeline(seed=42, difficulty="medium", category="FP")
        fmt = pipe.select_format()
        assert fmt != "csv"
        assert fmt in CATEGORY_FORMAT_MAP["FP"]

    def test_md_category_uses_md_corruptions(self, sample_df):
        pipe = CorruptionPipeline(seed=42, difficulty="easy", category="MD")
        pipe.select_format()
        dirty_df, error_map, sev, meta = pipe.corrupt(sample_df)
        applied = meta.get("corruptions_applied", [])
        md_set = set(CATEGORY_CORRUPTION_MAP["MD"])
        for name in applied:
            assert name in md_set

    def test_cp_category_uses_many_types(self, sample_df):
        pipe = CorruptionPipeline(seed=42, difficulty="hard", category="CP")
        pipe.select_format()
        dirty_df, error_map, sev, meta = pipe.corrupt(sample_df)
        applied = meta.get("corruptions_applied", [])
        assert len(applied) >= 7  # CP requires 7+

    def test_no_category_behaves_as_before(self, sample_df):
        pipe = CorruptionPipeline(seed=42, difficulty="medium")
        pipe.select_format()
        dirty_df, error_map, sev, meta = pipe.corrupt(sample_df)
        assert isinstance(dirty_df, pd.DataFrame)
        assert "corruptions_applied" in meta
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/shivanshu.s/deployments/proj/outlier_clean_data/.claude/worktrees/env-v2 && python -m pytest tests/test_categories.py::TestCategoryAwarePipeline -v`
Expected: FAIL — `__init__() got an unexpected keyword argument 'category'`

- [ ] **Step 3: Modify CorruptionPipeline to accept category**

In `server/corruption/pipeline.py`:

Update `__init__`:
```python
def __init__(self, seed: int, difficulty: Optional[str] = None, category: Optional[str] = None):
    self.seed = seed
    self.rng = np.random.default_rng(seed)
    self.py_rng = random.Random(seed)
    if difficulty is None:
        difficulty = self.py_rng.choices(
            list(DIFFICULTY_WEIGHTS.keys()),
            weights=list(DIFFICULTY_WEIGHTS.values()),
        )[0]
    self.difficulty = difficulty
    self.profile = DIFFICULTY_PROFILES[difficulty]
    self.category = category
    self._rules: list = []
```

Update `select_format()` to respect category:
```python
def select_format(self) -> str:
    if self.category:
        from .categories import get_formats_for_category
        pool = get_formats_for_category(self.category)
    else:
        pool = self.profile["format_pool"]
    return self.py_rng.choice(pool)
```

Update `corrupt()` to filter corruption types by category:
```python
def corrupt(
    self, clean_df: pd.DataFrame, rules: list | None = None,
) -> tuple[pd.DataFrame, dict, dict, dict]:
    self._rules = rules or []

    # Select corruption types
    if self.category:
        from .categories import get_corruptions_for_category
        allowed = get_corruptions_for_category(self.category)
        if allowed is None:
            # CP: use all, but ensure 7+ types
            pool = [entry for entry in CORRUPTION_REGISTRY]
            lo, hi = 7, min(10, len(pool))
        else:
            pool = [entry for entry in CORRUPTION_REGISTRY if entry["name"] in set(allowed)]
            lo = min(len(pool), self.profile["num_corruption_types"][0])
            hi = min(len(pool), self.profile["num_corruption_types"][1])
    else:
        pool = list(CORRUPTION_REGISTRY)
        lo, hi = self.profile["num_corruption_types"]

    n_types = self.rng.integers(lo, hi + 1) if lo < hi else lo
    n_types = min(n_types, len(pool))
    # ... rest of selection and application logic (use pool instead of full registry) ...
```

The key change: instead of sampling from the full `CORRUPTION_REGISTRY`, filter to `pool` based on category first. The rest of the corruption application logic stays the same.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/shivanshu.s/deployments/proj/outlier_clean_data/.claude/worktrees/env-v2 && python -m pytest tests/test_categories.py tests/test_corruption_pipeline.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add server/corruption/pipeline.py tests/test_categories.py
git commit -m "feat(pipeline): category-aware corruption selection constrains by FP/VR/MD/SR/SV/CP"
```

---

## Task 10: Environment — Category in Reset API

**Files:**
- Modify: `server/environment.py:230-327`

- [ ] **Step 1: Modify reset() to accept category**

In `server/environment.py`, update `reset()` to extract category from kwargs:

```python
def reset(
    self,
    seed: Optional[int] = None,
    episode_id: Optional[str] = None,
    **kwargs: Any,
) -> DataCleaningObservation:
    task_id = kwargs.get("task_id")
    difficulty = kwargs.get("difficulty", "medium")
    category = kwargs.get("category")  # NEW: optional category constraint
```

Pass category to CorruptionPipeline:

```python
pipeline = CorruptionPipeline(seed=seed or 42, difficulty=difficulty, category=category)
```

- [ ] **Step 2: Run full test suite**

Run: `cd /Users/shivanshu.s/deployments/proj/outlier_clean_data/.claude/worktrees/env-v2 && python -m pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 3: Commit**

```bash
git add server/environment.py
git commit -m "feat(env): accept category param in reset() for targeted benchmarking"
```

---

## Task 11: Run Catalog Enricher + Verify

**Files:**
- Modify: `datasets/catalog.json` (auto-generated)

- [ ] **Step 1: Run the enricher on the real catalog**

```bash
cd /Users/shivanshu.s/deployments/proj/outlier_clean_data/.claude/worktrees/env-v2
python -m server.rules.catalog_enricher datasets/catalog.json datasets/clean/
```

Expected: stdout shows "Enriched N datasets" with some warnings for missing CSVs.

- [ ] **Step 2: Verify catalog has rules**

```bash
python -c "
import json
with open('datasets/catalog.json') as f:
    catalog = json.load(f)
with_rules = [e for e in catalog if 'rules' in e]
print(f'{len(with_rules)}/{len(catalog)} datasets enriched with rules')
if with_rules:
    print(f'Example rules for {with_rules[0][\"id\"]}:')
    for r in with_rules[0]['rules'][:3]:
        print(f'  {r}')
"
```

Expected: Most datasets with clean CSVs should have rules.

- [ ] **Step 3: Commit enriched catalog**

```bash
git add datasets/catalog.json
git commit -m "data(catalog): auto-enrich all datasets with inferred semantic rules"
```

---

## Task 12: Update .context Documentation

**Files:**
- Modify: `.context/architecture.md`
- Modify: `.context/PROJECT.md`

- [ ] **Step 1: Update architecture.md**

Add a section for the rules subsystem and category system:

```markdown
### Semantic Rules (`server/rules/`)

Seven rule types (range, regex, enum, dtype, not_null, unique, cross_column)
auto-inferred from clean data and stored in `catalog.json`. At reset(),
rules are loaded and passed to the corruption pipeline (for targeted violations)
and to the grader (5th scoring dimension: semantic 10%).

### Benchmark Categories (`server/corruption/categories.py`)

Six categories (FP, VR, MD, SR, SV, CP) map to subsets of the 22 corruption
types. The pipeline accepts an optional `category` param to constrain corruption
selection for targeted benchmarking.
```

- [ ] **Step 2: Update PROJECT.md with new grading weights**

Update the grading formula section to reflect 5-dimension scoring:
- Schema 15%, Row 15%, Cell 50%, Distribution 10%, Semantic 10%

- [ ] **Step 3: Commit**

```bash
git add .context/
git commit -m "docs(.context): update architecture and project docs for rules + categories"
```

---

## Task 13: Final Integration Test

- [ ] **Step 1: Run full test suite**

```bash
cd /Users/shivanshu.s/deployments/proj/outlier_clean_data/.claude/worktrees/env-v2
python -m pytest tests/ -v --tb=short
```

Expected: All tests PASS.

- [ ] **Step 2: Smoke test end-to-end**

```bash
python -c "
from server.environment import DataCleaningEnvironment
env = DataCleaningEnvironment()
obs = env.reset(seed=42, task_id='titanic', difficulty='medium', category='VR')
print(f'Task: {obs.task_id}')
print(f'Rules: {len(obs.semantic_rules)} rules loaded')
print(f'Format: {obs.file_format}')
print(f'Reward: {obs.step_info.reward}')
for r in obs.semantic_rules[:3]:
    print(f'  Rule: {r}')
"
```

Expected: Environment resets with VR category, loads titanic rules, shows initial reward.

- [ ] **Step 3: Verify backward compatibility — no category**

```bash
python -c "
from server.environment import DataCleaningEnvironment
env = DataCleaningEnvironment()
obs = env.reset(seed=42, task_id='titanic', difficulty='easy')
print(f'No-category reset: reward={obs.step_info.reward}, rules={len(obs.semantic_rules)}')
"
```

Expected: Works exactly as before, rules loaded but grading uses legacy weights if rules are empty.
