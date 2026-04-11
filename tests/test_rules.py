import pytest
import pandas as pd

try:
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
    from server.rules.inferrer import infer_rules
    from server.rules.validator import Violation, compute_semantic_score, validate
except ImportError:
    from rules.types import (
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
    from rules.inferrer import infer_rules
    from rules.validator import Violation, compute_semantic_score, validate


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


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "age": [25, 35, -5, 200, 45],
        "email": ["a@b.com", "bad", "c@d.org", "e@f.io", ""],
        "sex": ["male", "female", "male", "alien", "female"],
        "score": [88.5, None, 75.0, 90.0, None],
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
        assert all(isinstance(v, Violation) for v in violations)
        assert all(v.rule_type == "range" for v in violations)
        assert {v.row_index for v in violations} == {2, 3}

    def test_validates_regex_violations(self, sample_df, sample_rules):
        violations = validate(sample_df, [sample_rules[1]])
        assert len(violations) == 2
        assert all(v.rule_type == "regex" for v in violations)

    def test_validates_enum_violations(self, sample_df, sample_rules):
        violations = validate(sample_df, [sample_rules[2]])
        assert len(violations) == 1
        assert violations[0].row_index == 3
        assert violations[0].actual_value == "alien"

    def test_validates_enum_case_insensitively(self):
        df = pd.DataFrame({"sex": ["MALE", "Female", "other", "alien"]})
        rule = EnumRule(column="sex", values=["male", "female", "other"], source="statistical")
        violations = validate(df, [rule])
        assert len(violations) == 1
        assert violations[0].actual_value == "alien"

    def test_validates_not_null_violations(self, sample_df, sample_rules):
        violations = validate(sample_df, [sample_rules[3]])
        assert len(violations) == 2
        assert all(v.rule_type == "not_null" for v in violations)

    def test_validates_unique_rule(self):
        df = pd.DataFrame({"id": [1, 2, 3, 2, 5]})
        violations = validate(df, [UniqueRule(column="id", source="heuristic")])
        assert len(violations) >= 1
        assert all(v.rule_type == "unique" for v in violations)

    def test_validates_dtype_rule(self):
        df = pd.DataFrame({"age": [25, "thirty", 40, 50]})
        violations = validate(
            df,
            [DtypeRule(column="age", expected_dtype="integer", source="statistical")],
        )
        assert len(violations) >= 1
        assert violations[0].actual_value == "thirty"

    def test_all_rules_combined(self, sample_df, sample_rules):
        violations = validate(sample_df, sample_rules)
        assert len(violations) == 7

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
        assert validate(df, rules) == []
        assert compute_semantic_score(df, rules) == 1.0

    def test_semantic_score_penalizes_violations(self, sample_df, sample_rules):
        score = compute_semantic_score(sample_df, sample_rules)
        assert score == pytest.approx(1.0 - 7 / 20)

    def test_semantic_score_returns_perfect_when_no_rules_apply(self):
        df = pd.DataFrame({"age": [25, 35, 45]})
        rules = [RangeRule(column="missing", min_val=0, max_val=120, source="heuristic")]
        assert compute_semantic_score(df, rules) == 1.0

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
        assert violations[0].actual_value == "WRONG"

    def test_cross_column_ordering(self):
        df = pd.DataFrame({
            "start": [1, 4, 3],
            "end": [2, 3, 3],
        })
        rule = CrossColumnRule(
            columns=["start", "end"],
            condition="ordering",
            hint="start must be less than or equal to end",
            source="manual",
        )
        violations = validate(df, [rule])
        assert len(violations) == 1
        assert violations[0].row_index == 1
        assert violations[0].actual_value == "4 > 3"

    def test_cross_column_skips_missing_columns(self):
        df = pd.DataFrame({"start": [1, 2, 3]})
        rule = CrossColumnRule(
            columns=["start", "end"],
            condition="ordering",
            hint="start must be less than or equal to end",
            source="manual",
        )
        assert validate(df, [rule]) == []


class TestInferrerDomainHeuristics:
    def test_age_column_gets_range_rule(self):
        df = pd.DataFrame({"age": [25, 35, 45, 55]})
        rules = infer_rules(df)
        range_rules = [
            rule
            for rule in rules
            if rule.rule_type == "range" and rule.column == "age"
        ]
        assert len(range_rules) == 1
        assert range_rules[0].min_val == 0
        assert range_rules[0].max_val == 120
        assert range_rules[0].source == "heuristic"

    def test_email_column_gets_regex_rule(self):
        df = pd.DataFrame({"email": ["a@b.com", "c@d.org"]})
        rules = infer_rules(df)
        regex_rules = [
            rule
            for rule in rules
            if rule.rule_type == "regex" and rule.column == "email"
        ]
        assert len(regex_rules) == 1
        assert regex_rules[0].source == "heuristic"

    def test_id_column_gets_unique_and_not_null(self):
        df = pd.DataFrame({"passenger_id": [1, 2, 3]})
        rules = infer_rules(df)
        unique_rules = [
            rule
            for rule in rules
            if rule.rule_type == "unique" and rule.column == "passenger_id"
        ]
        not_null_rules = [
            rule
            for rule in rules
            if rule.rule_type == "not_null" and rule.column == "passenger_id"
        ]
        assert len(unique_rules) == 1
        assert len(not_null_rules) == 1

    def test_salary_column_gets_nonneg_range(self):
        df = pd.DataFrame({"salary": [50000, 80000, 120000]})
        rules = infer_rules(df)
        range_rules = [
            rule
            for rule in rules
            if rule.rule_type == "range" and rule.column == "salary"
        ]
        assert len(range_rules) == 1
        assert range_rules[0].min_val == 0
        assert range_rules[0].max_val == float("inf")
        assert range_rules[0].source == "heuristic"


class TestInferrerStatistical:
    def test_numeric_column_gets_expanded_range(self):
        df = pd.DataFrame({"score": [10.0, 20.0, 30.0, 40.0, 50.0]})
        rules = infer_rules(df)
        range_rules = [
            rule
            for rule in rules
            if rule.rule_type == "range" and rule.column == "score"
        ]
        assert len(range_rules) == 1
        rule = range_rules[0]
        assert rule.min_val == 0
        assert rule.max_val == pytest.approx(58.0)
        assert rule.source == "statistical"

    def test_categorical_column_gets_enum(self):
        df = pd.DataFrame({"color": ["red", "blue", "green", "red", "blue"]})
        rules = infer_rules(df)
        enum_rules = [
            rule
            for rule in rules
            if rule.rule_type == "enum" and rule.column == "color"
        ]
        assert len(enum_rules) == 1
        assert set(enum_rules[0].values) == {"red", "blue", "green"}

    def test_high_cardinality_column_no_enum(self):
        df = pd.DataFrame({"name": [f"person_{i}" for i in range(50)]})
        rules = infer_rules(df)
        enum_rules = [
            rule
            for rule in rules
            if rule.rule_type == "enum" and rule.column == "name"
        ]
        assert len(enum_rules) == 0

    def test_no_null_column_gets_not_null_rule(self):
        df = pd.DataFrame({"val": [1, 2, 3, 4, 5]})
        rules = infer_rules(df)
        not_null_rules = [
            rule
            for rule in rules
            if rule.rule_type == "not_null" and rule.column == "val"
        ]
        assert len(not_null_rules) == 1

    def test_column_with_nulls_no_not_null_rule(self):
        df = pd.DataFrame({"val": [1, None, 3, None, 5]})
        rules = infer_rules(df)
        not_null_rules = [
            rule
            for rule in rules
            if rule.rule_type == "not_null" and rule.column == "val"
        ]
        assert len(not_null_rules) == 0


class TestInferrerCrossColumn:
    def test_detects_functional_dependency(self):
        df = pd.DataFrame({
            "code": ["A", "B", "A", "C", "B", "A", "C"],
            "name": [
                "Alpha",
                "Beta",
                "Alpha",
                "Charlie",
                "Beta",
                "Alpha",
                "Charlie",
            ],
        })
        rules = infer_rules(df)
        cross_column_rules = [
            rule for rule in rules if rule.rule_type == "cross_column"
        ]
        assert len(cross_column_rules) >= 1
        assert any(
            "code" in rule.columns and "name" in rule.columns
            for rule in cross_column_rules
        )

    def test_no_false_dependency_on_numeric_columns(self):
        df = pd.DataFrame({
            "x": [1, 2, 3, 4, 5],
            "y": [10, 20, 30, 40, 50],
        })
        rules = infer_rules(df)
        cross_column_rules = [
            rule for rule in rules if rule.rule_type == "cross_column"
        ]
        assert len(cross_column_rules) == 0
