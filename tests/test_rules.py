import pytest

from server.rules.types import (
    CrossColumnRule,
    DtypeRule,
    EnumRule,
    NotNullRule,
    RangeRule,
    RegexRule,
    UniqueRule,
    rule_from_dict,
    rule_to_dict,
)


class TestRuleTypes:
    def test_range_rule_creation(self):
        rule = RangeRule(column="age", min_val=0, max_val=120, source="heuristic")
        assert rule.column == "age"
        assert rule.min_val == 0
        assert rule.max_val == 120
        assert rule.rule_type == "range"

    def test_regex_rule_creation(self):
        rule = RegexRule(
            column="email",
            pattern=r"^[\w.]+@[\w.]+\.\w+$",
            source="heuristic",
        )
        assert rule.rule_type == "regex"
        assert rule.pattern == r"^[\w.]+@[\w.]+\.\w+$"

    def test_enum_rule_creation(self):
        rule = EnumRule(column="sex", values=["male", "female"], source="statistical")
        assert rule.rule_type == "enum"
        assert set(rule.values) == {"male", "female"}

    def test_dtype_rule_creation(self):
        rule = DtypeRule(column="age", expected_dtype="integer", source="statistical")
        assert rule.rule_type == "dtype"
        assert rule.expected_dtype == "integer"

    def test_not_null_rule_creation(self):
        rule = NotNullRule(column="id", source="statistical")
        assert rule.rule_type == "not_null"

    def test_unique_rule_creation(self):
        rule = UniqueRule(column="id", source="heuristic")
        assert rule.rule_type == "unique"

    def test_cross_column_rule_creation(self):
        rule = CrossColumnRule(
            columns=["embarked", "port"],
            condition="functional_dependency",
            hint="S=Southampton, C=Cherbourg",
            source="cross_column_inference",
        )
        assert rule.rule_type == "cross_column"
        assert rule.hint == "S=Southampton, C=Cherbourg"


class TestRuleSerialization:
    @pytest.mark.parametrize(
        ("original", "expected_dict"),
        [
            (
                RangeRule(column="age", min_val=0, max_val=120, source="manual"),
                {
                    "type": "range",
                    "column": "age",
                    "min": 0,
                    "max": 120,
                    "source": "manual",
                },
            ),
            (
                RegexRule(
                    column="email",
                    pattern=r"^[\w.]+@[\w.]+\.\w+$",
                    source="heuristic",
                ),
                {
                    "type": "regex",
                    "column": "email",
                    "pattern": r"^[\w.]+@[\w.]+\.\w+$",
                    "source": "heuristic",
                },
            ),
            (
                EnumRule(
                    column="sex",
                    values=["male", "female"],
                    source="statistical",
                ),
                {
                    "type": "enum",
                    "column": "sex",
                    "values": ["male", "female"],
                    "source": "statistical",
                },
            ),
            (
                DtypeRule(
                    column="age",
                    expected_dtype="integer",
                    source="statistical",
                ),
                {
                    "type": "dtype",
                    "column": "age",
                    "expected_dtype": "integer",
                    "source": "statistical",
                },
            ),
            (
                NotNullRule(column="id", source="statistical"),
                {
                    "type": "not_null",
                    "column": "id",
                    "source": "statistical",
                },
            ),
            (
                UniqueRule(column="id", source="heuristic"),
                {
                    "type": "unique",
                    "column": "id",
                    "source": "heuristic",
                },
            ),
            (
                CrossColumnRule(
                    columns=["city", "state"],
                    condition="functional_dependency",
                    hint="City determines state",
                    source="cross_column_inference",
                ),
                {
                    "type": "cross_column",
                    "columns": ["city", "state"],
                    "condition": "functional_dependency",
                    "hint": "City determines state",
                    "source": "cross_column_inference",
                },
            ),
        ],
    )
    def test_round_trip(self, original, expected_dict):
        serialized = rule_to_dict(original)
        assert serialized == expected_dict
        assert rule_from_dict(serialized) == original

    def test_from_dict_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown rule type"):
            rule_from_dict({"type": "bogus", "column": "x"})
