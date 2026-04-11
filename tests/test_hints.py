"""Tests for the 3-level hint generation system."""
from __future__ import annotations

import importlib.util
import sys
import os
import pytest

# Load hints.py directly by file path to avoid triggering
# server/corruption/__init__.py which may import sibling modules
# (pipeline, value_corruptions) not yet created by parallel tasks.
_ROOT = os.path.join(os.path.dirname(__file__), "..")
_HINTS_PATH = os.path.join(_ROOT, "server", "corruption", "hints.py")
_spec = importlib.util.spec_from_file_location("hints", _HINTS_PATH)
_hints = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_hints)

generate_hints = _hints.generate_hints
generate_format_hints = _hints.generate_format_hints
HINT_TEMPLATES = _hints.HINT_TEMPLATES
_approx = _hints._approx


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def null_error_map():
    """Error map with inject_nulls errors in two columns."""
    return {
        "cell_errors": {
            "0,Age": {"severity": 2.0, "clean_value": 29.0, "corruption": "inject_nulls"},
            "1,Age": {"severity": 2.0, "clean_value": 35.0, "corruption": "inject_nulls"},
            "2,Age": {"severity": 2.0, "clean_value": 22.0, "corruption": "inject_nulls"},
            "3,Age": {"severity": 2.0, "clean_value": 40.0, "corruption": "inject_nulls"},
            "4,Age": {"severity": 2.0, "clean_value": 31.0, "corruption": "inject_nulls"},
            "5,Fare": {"severity": 2.0, "clean_value": 7.25, "corruption": "inject_nulls"},
            "6,Fare": {"severity": 2.0, "clean_value": 13.0, "corruption": "inject_nulls"},
            "7,Fare": {"severity": 2.0, "clean_value": 8.05, "corruption": "inject_nulls"},
        },
        "spurious_rows": {},
        "missing_rows": {},
    }


@pytest.fixture
def col_stats():
    return {
        "Age": {"mean": 30.0, "median": 28.0},
        "Fare": {"mean": 10.0, "median": 8.0},
    }


@pytest.fixture
def spurious_error_map():
    return {
        "cell_errors": {},
        "spurious_rows": {
            "5": {"severity": 2.0},
            "10": {"severity": 2.0},
        },
        "missing_rows": {},
    }


@pytest.fixture
def missing_error_map():
    return {
        "cell_errors": {},
        "spurious_rows": {},
        "missing_rows": {
            "missing_0": {"severity": 2.5, "clean_values": {"Name": "Alice", "Age": 25}},
            "missing_1": {"severity": 2.5, "clean_values": {"Name": "Bob", "Age": 30}},
            "missing_2": {"severity": 2.5, "clean_values": {"Name": "Carol", "Age": 35}},
        },
    }


@pytest.fixture
def format_metadata():
    return [
        {"type": "delimiter_change", "details": "Tabs replaced commas"},
        {"type": "encoding_shift", "details": "UTF-16 instead of UTF-8"},
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestStrategyHints:
    def test_strategy_hints_include_code(self, null_error_map, col_stats):
        """Strategy level should mention code suggestions like fillna/mean."""
        hints = generate_hints(null_error_map, "strategy", col_stats)
        assert "fillna" in hints.lower() or "mean" in hints.lower()
        # Should also include column stats
        assert "30.0" in hints
        assert "28.0" in hints

    def test_strategy_shows_columns(self, null_error_map, col_stats):
        hints = generate_hints(null_error_map, "strategy", col_stats)
        assert "Age" in hints
        assert "Fare" in hints


class TestTacticalHints:
    def test_tactical_hints_show_columns_and_counts(self, null_error_map):
        hints = generate_hints(null_error_map, "tactical")
        assert "Age" in hints
        assert "Fare" in hints
        assert "(5)" in hints  # 5 Age nulls
        assert "(3)" in hints  # 3 Fare nulls

    def test_tactical_no_code_suggestions(self, null_error_map):
        hints = generate_hints(null_error_map, "tactical")
        assert "fillna" not in hints.lower()


class TestCategoricalHints:
    def test_categorical_hints_are_vague(self, null_error_map):
        """Categorical should not reveal column names."""
        hints = generate_hints(null_error_map, "categorical")
        assert "Age" not in hints
        assert "Fare" not in hints

    def test_categorical_uses_approximate_count(self, null_error_map):
        hints = generate_hints(null_error_map, "categorical")
        # 8 total errors -> _approx(8) = 10
        assert "~10" in hints or "~5" in hints


class TestFormatHints:
    def test_format_hints_strategy(self, format_metadata):
        hints = generate_format_hints(format_metadata, "strategy")
        assert "delimiter_change" in hints
        assert "Tabs replaced commas" in hints
        assert "encoding_shift" in hints

    def test_format_hints_tactical(self, format_metadata):
        hints = generate_format_hints(format_metadata, "tactical")
        assert "delimiter_change" in hints
        # Tactical should not include details
        assert "Tabs replaced commas" not in hints

    def test_format_hints_categorical_vague(self, format_metadata):
        hints = generate_format_hints(format_metadata, "categorical")
        lines = [l for l in hints.strip().splitlines() if l.strip()]
        assert len(lines) == 1
        assert "format" in hints.lower() or "issue" in hints.lower()

    def test_format_hints_empty(self):
        assert generate_format_hints([], "strategy") == ""


class TestTemplateRegistry:
    def test_all_22_corruptions_have_templates(self):
        expected = {
            "inject_nulls", "type_mangle", "duplicate_rows", "whitespace_noise",
            "format_inconsistency", "outlier_injection", "drop_rows",
            "decimal_shift", "value_swap", "typo_injection", "date_format_mix",
            "abbreviation_mix", "leading_zero_strip", "header_in_data",
            "category_misspell", "business_rule_violation", "encoding_noise",
            "schema_drift", "unicode_homoglyph", "html_entity_leak",
            "column_shift", "unit_inconsistency",
        }
        assert expected == set(HINT_TEMPLATES.keys())
        assert len(HINT_TEMPLATES) == 22

    def test_each_template_callable(self):
        for name, fn in HINT_TEMPLATES.items():
            assert callable(fn), f"{name} template is not callable"

    def test_each_template_returns_string_at_all_levels(self):
        """Every template must return a non-empty string for all 3 levels."""
        cols = {"TestCol": 4}
        for name, fn in HINT_TEMPLATES.items():
            for level in ("strategy", "tactical", "categorical"):
                result = fn(name, cols, 4, level, None)
                assert isinstance(result, str), f"{name}/{level} did not return str"
                assert len(result) > 0, f"{name}/{level} returned empty string"


class TestSpuriousRows:
    def test_spurious_rows_hint_strategy(self, spurious_error_map):
        hints = generate_hints(spurious_error_map, "strategy")
        assert "spurious" in hints.lower() or "extra" in hints.lower()
        assert "5" in hints
        assert "10" in hints
        assert "drop" in hints.lower()

    def test_spurious_rows_hint_tactical(self, spurious_error_map):
        hints = generate_hints(spurious_error_map, "tactical")
        assert "2" in hints  # 2 spurious rows
        assert "spurious" in hints.lower() or "extra" in hints.lower()

    def test_spurious_rows_hint_categorical(self, spurious_error_map):
        hints = generate_hints(spurious_error_map, "categorical")
        assert "extra" in hints.lower() or "spurious" in hints.lower()
        # Should not contain exact row indices
        assert "5," not in hints


class TestMissingRows:
    def test_missing_rows_hint_strategy(self, missing_error_map):
        hints = generate_hints(missing_error_map, "strategy")
        assert "missing" in hints.lower()
        assert "3" in hints

    def test_missing_rows_hint_tactical(self, missing_error_map):
        hints = generate_hints(missing_error_map, "tactical")
        assert "missing" in hints.lower()
        assert "3" in hints

    def test_missing_rows_hint_categorical(self, missing_error_map):
        hints = generate_hints(missing_error_map, "categorical")
        assert "missing" in hints.lower()


class TestEmptyErrorMap:
    def test_empty_error_map(self):
        empty = {"cell_errors": {}, "spurious_rows": {}, "missing_rows": {}}
        hints = generate_hints(empty, "strategy")
        assert hints == ""

    def test_completely_empty_dict(self):
        hints = generate_hints({}, "tactical")
        assert hints == ""


class TestApprox:
    def test_approx_rounds_to_nearest_five(self):
        assert _approx(8) == 10
        assert _approx(12) == 10
        assert _approx(13) == 15
        assert _approx(3) == 5  # minimum is 5
        assert _approx(0) == 0
