"""Tests for multi-level grader."""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from server.grader import (
    grade, schema_score, row_score, cell_score, distribution_score,
    match_rows_by_content, _values_equal, _dtypes_compatible,
    _detect_collateral_damage, _check_spurious_row,
)

@pytest.fixture
def clean_df():
    return pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
        "score": [88.5, 92.0, 76.3, 95.1, 81.0],
        "active": [True, False, True, True, False],
    })

@pytest.fixture
def error_map_cell():
    return {
        "cell_errors": {
            "0,score": {"severity": 1.0, "clean_value": 88.5, "corruption": "outlier_injection", "accepted_fill": None},
            "2,name": {"severity": 1.0, "clean_value": "Charlie", "corruption": "typo_noise", "accepted_fill": None},
        },
        "spurious_rows": {},
        "missing_rows": {},
    }

class TestValuesEqual:
    def test_equal_ints(self): assert _values_equal(1, 1)
    def test_equal_floats(self): assert _values_equal(1.0, 1.0)
    def test_int_float(self): assert _values_equal(1, 1.0)
    def test_nan_nan(self): assert _values_equal(float("nan"), float("nan"))
    def test_nan_none(self): assert _values_equal(None, float("nan"))
    def test_not_equal(self): assert not _values_equal(1, 2)

class TestSchemaScore:
    def test_perfect(self, clean_df):
        assert schema_score(clean_df, clean_df) == pytest.approx(1.0)

    def test_missing_column(self, clean_df):
        result = clean_df.drop(columns=["score"])
        s = schema_score(clean_df, result)
        assert s < 1.0

    def test_extra_column_no_penalty(self, clean_df):
        result = clean_df.copy()
        result["extra"] = 0
        s = schema_score(clean_df, result)
        assert s == pytest.approx(1.0)

    def test_case_insensitive(self, clean_df):
        result = clean_df.rename(columns=str.upper)
        s = schema_score(clean_df, result)
        assert s >= 0.7  # names match case-insensitively

    def test_empty_clean(self):
        assert schema_score(pd.DataFrame(), pd.DataFrame()) == 1.0

class TestMatchRowsByContent:
    def test_perfect_match(self, clean_df):
        mapping = match_rows_by_content(clean_df, clean_df)
        assert len(mapping) == len(clean_df)

    def test_shuffled(self, clean_df):
        shuffled = clean_df.sample(frac=1, random_state=42).reset_index(drop=True)
        mapping = match_rows_by_content(clean_df, shuffled)
        assert len(mapping) == len(clean_df)

    def test_missing_row(self, clean_df):
        result = clean_df.iloc[1:].reset_index(drop=True)
        mapping = match_rows_by_content(clean_df, result)
        assert len(mapping) == len(clean_df) - 1

class TestRowScore:
    def test_perfect(self, clean_df):
        s = row_score(clean_df, clean_df, {"spurious_rows": {}, "missing_rows": {}})
        assert s >= 0.8

    def test_missing_rows(self, clean_df):
        result = clean_df.iloc[:3].reset_index(drop=True)
        error_map = {
            "spurious_rows": {},
            "missing_rows": {"3": {"severity": 1.0}, "4": {"severity": 1.0}},
        }
        s = row_score(clean_df, result, error_map)
        assert s < 1.0

    def test_spurious_row_still_present_even_if_result_trimmed(self):
        clean = pd.DataFrame({"id": [1, 2], "name": ["Alice", "Bob"]})
        result = pd.DataFrame(
            {"id": [1, 999], "name": ["Alice", "Alice"]},
            index=[0, 2],
        )

        assert not _check_spurious_row(clean, result, "2")

    def test_spurious_negative_row_id_is_not_fixed(self, clean_df):
        assert not _check_spurious_row(clean_df, clean_df, "-1")

class TestCellScore:
    def test_perfect(self, clean_df, error_map_cell):
        mapping = match_rows_by_content(clean_df, clean_df)
        s = cell_score(clean_df, clean_df, error_map_cell, mapping)
        assert s == pytest.approx(1.0)

    def test_unfixed(self, clean_df, error_map_cell):
        dirty = clean_df.copy()
        dirty.at[0, "score"] = 999.0  # corruption still present
        mapping = match_rows_by_content(clean_df, dirty)
        s = cell_score(clean_df, dirty, error_map_cell, mapping)
        assert s < 1.0

    def test_no_errors(self, clean_df):
        mapping = match_rows_by_content(clean_df, clean_df)
        s = cell_score(clean_df, clean_df, {"cell_errors": {}}, mapping)
        assert s == pytest.approx(1.0)

    def test_collateral_damage_uses_row_mapping_for_reordered_rows(self):
        clean = pd.DataFrame({"id": [1, 2, 3], "name": ["Alice", "Bob", "Cara"]})
        result = clean.iloc[[2, 0, 1]].reset_index(drop=True)
        mapping = match_rows_by_content(clean, result)

        collateral = _detect_collateral_damage(
            clean,
            result,
            {"cell_errors": {}},
            mapping,
        )

        assert collateral == pytest.approx(0.0)

    def test_collateral_damage_skips_dropped_rows_missing_from_mapping(self):
        clean = pd.DataFrame({"id": [1, 2, 3], "name": ["Alice", "Bob", "Cara"]})
        result = clean.drop(index=1).reset_index(drop=True)
        mapping = match_rows_by_content(clean, result)

        collateral = _detect_collateral_damage(
            clean,
            result,
            {"cell_errors": {}},
            mapping,
        )

        assert collateral == pytest.approx(0.0)

    @pytest.mark.parametrize("mapping", [None, {}])
    def test_collateral_damage_falls_back_to_clean_index_without_mapping(self, mapping):
        clean = pd.DataFrame({"id": [1, 2], "name": ["Alice", "Bob"]})
        result = clean.copy()
        result.at[1, "name"] = "Bobby"

        collateral = _detect_collateral_damage(
            clean,
            result,
            {"cell_errors": {}},
            mapping,
        )

        assert collateral == pytest.approx(0.5)

class TestDistributionScore:
    def test_perfect(self, clean_df):
        s = distribution_score(clean_df, clean_df, {"score"})
        assert s == pytest.approx(1.0)

    def test_imputed_close(self, clean_df):
        result = clean_df.copy()
        result.at[0, "score"] = clean_df["score"].mean()  # imputed with mean
        s = distribution_score(clean_df, result, {"score"})
        assert s > 0.7

    def test_no_imputed_cols(self, clean_df):
        s = distribution_score(clean_df, clean_df, set())
        assert s == pytest.approx(1.0)

class TestGrade:
    def test_perfect_score(self, clean_df):
        error_map = {"cell_errors": {}, "spurious_rows": {}, "missing_rows": {}}
        _, reward, *_ = grade(clean_df, clean_df, error_map, 2, 2, 10)
        assert reward >= 0.9

    def test_reward_in_range(self, clean_df, error_map_cell):
        _, reward, *_ = grade(clean_df, clean_df, error_map_cell, 3, 2, 10)
        assert 0.0 <= reward <= 1.0

    def test_undo_cost(self, clean_df):
        error_map = {"cell_errors": {}, "spurious_rows": {}, "missing_rows": {}}
        _, r1, *_ = grade(clean_df, clean_df, error_map, 2, 2, 10, undo_count=0)
        _, r2, *_ = grade(clean_df, clean_df, error_map, 2, 2, 10, undo_count=5)
        assert r1 > r2

    def test_grade_passes_row_mapping_to_cell_scoring(self):
        clean = pd.DataFrame({"id": [1, 2], "name": ["Alice", "Bob"]})
        result = clean.iloc[[1, 0]].reset_index(drop=True)
        error_map = {
            "cell_errors": {
                "0,name": {
                    "severity": 1.0,
                    "clean_value": "Alice",
                    "corruption": "typo_noise",
                    "accepted_fill": None,
                },
            },
            "spurious_rows": {},
            "missing_rows": {},
        }

        error_status, _, _, row_mapping = grade(clean, result, error_map, 2, 2, 10)

        assert row_mapping[0] == 1
        assert error_status["0,name"] == "fixed"


def test_performance_20k_rows():
    import time
    big = pd.DataFrame({
        "a": range(20000),
        "b": [float(i) for i in range(20000)],
        "c": ["x"] * 20000,
    })
    error_map = {"cell_errors": {}, "spurious_rows": {}, "missing_rows": {}}
    t0 = time.time()
    grade(big, big, error_map, 2, 2, 10)
    elapsed = time.time() - t0
    assert elapsed < 5.0, f"grade() took {elapsed:.1f}s on 20K rows"
