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
        _, reward, *_ = grade(
            clean_df, clean_df, error_map,
            transform_steps=3, min_transform_steps=2, max_transform_steps=10,
            rules=rules,
        )
        assert reward > 0.9

    def test_rule_violations_reduce_score(self, clean_df, error_map, rules):
        bad_result = clean_df.copy()
        bad_result.loc[0, "age"] = -5
        bad_result.loc[1, "sex"] = "alien"

        _, reward_bad, *_ = grade(
            clean_df, bad_result, error_map,
            transform_steps=3, min_transform_steps=2, max_transform_steps=10,
            rules=rules,
        )
        _, reward_good, *_ = grade(
            clean_df, clean_df, error_map,
            transform_steps=3, min_transform_steps=2, max_transform_steps=10,
            rules=rules,
        )
        assert reward_good > reward_bad

    def test_no_rules_defaults_to_full_score(self, clean_df, error_map):
        _, reward_no_rules, *_ = grade(
            clean_df, clean_df, error_map,
            transform_steps=3, min_transform_steps=2, max_transform_steps=10,
        )
        _, reward_empty_rules, *_ = grade(
            clean_df, clean_df, error_map,
            transform_steps=3, min_transform_steps=2, max_transform_steps=10,
            rules=[],
        )
        assert reward_no_rules == pytest.approx(reward_empty_rules)

    def test_reward_clamped_0_to_1(self, clean_df, error_map, rules):
        terrible = clean_df.copy()
        terrible["age"] = [-999, -999, -999, -999, -999]
        terrible["sex"] = ["x", "x", "x", "x", "x"]
        terrible["score"] = [None, None, None, None, None]

        _, reward, *_ = grade(
            clean_df, terrible, error_map,
            transform_steps=3, min_transform_steps=2, max_transform_steps=10,
            rules=rules,
        )
        assert 0.0 <= reward <= 1.0
