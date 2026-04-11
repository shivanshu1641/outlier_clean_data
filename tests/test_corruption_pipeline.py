"""Tests for the runtime corruption pipeline."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from server.corruption import (
    CORRUPTION_REGISTRY,
    CORRUPTION_SEVERITY,
    DIFFICULTY_PROFILES,
    DIFFICULTY_WEIGHTS,
    CorruptionPipeline,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Small mixed-type DataFrame for basic corruption tests."""
    rng = np.random.default_rng(0)
    n = 100
    return pd.DataFrame({
        "id": [f"{i:04d}" for i in range(n)],
        "name": [f"Person_{i}" for i in range(n)],
        "age": rng.integers(18, 80, size=n).tolist(),
        "salary": (rng.random(n) * 100_000 + 30_000).round(2).tolist(),
        "city": rng.choice(["New York", "California", "Texas", "Florida", "Ohio"], size=n).tolist(),
        "date_joined": pd.date_range("2020-01-01", periods=n).strftime("%Y-%m-%d").tolist(),
        "active": rng.choice(["Yes", "No"], size=n).tolist(),
    })


@pytest.fixture
def large_df() -> pd.DataFrame:
    """5K-row DataFrame for performance testing."""
    rng = np.random.default_rng(99)
    n = 5000
    return pd.DataFrame({
        "id": [f"{i:06d}" for i in range(n)],
        "name": [f"User_{i}" for i in range(n)],
        "score": rng.random(n).round(4).tolist(),
        "amount": (rng.random(n) * 10_000).round(2).tolist(),
        "category": rng.choice(["Alpha", "Beta", "Gamma", "Delta"], size=n).tolist(),
        "region": rng.choice(["North", "South", "East", "West"], size=n).tolist(),
    })


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCorruptionPipelineBasic:
    """Core pipeline behavior."""

    def test_pipeline_produces_dirty_df_and_error_map(self, sample_df: pd.DataFrame):
        pipe = CorruptionPipeline(seed=42, difficulty="medium")
        fmt = pipe.select_format()
        dirty_df, error_map, severity_map, meta = pipe.corrupt(sample_df)

        assert isinstance(dirty_df, pd.DataFrame)
        assert isinstance(error_map, dict)
        assert "cell_errors" in error_map
        assert "spurious_rows" in error_map
        assert "missing_rows" in error_map
        assert isinstance(severity_map, dict)
        assert "total_severity" in severity_map
        assert "by_type" in severity_map
        assert isinstance(meta, dict)
        assert meta["difficulty"] == "medium"
        assert meta["seed"] == 42
        assert isinstance(fmt, str)

    def test_pipeline_reproducible(self, sample_df: pd.DataFrame):
        """Same seed must produce identical output."""
        pipe1 = CorruptionPipeline(seed=123, difficulty="easy")
        fmt1 = pipe1.select_format()
        dirty1, emap1, smap1, meta1 = pipe1.corrupt(sample_df)

        pipe2 = CorruptionPipeline(seed=123, difficulty="easy")
        fmt2 = pipe2.select_format()
        dirty2, emap2, smap2, meta2 = pipe2.corrupt(sample_df)

        assert fmt1 == fmt2
        pd.testing.assert_frame_equal(dirty1, dirty2)
        assert emap1 == emap2
        assert smap1 == smap2
        assert meta1 == meta2

    def test_pipeline_different_seeds_different_output(self, sample_df: pd.DataFrame):
        """Different seeds should produce different corruptions."""
        pipe1 = CorruptionPipeline(seed=1, difficulty="medium")
        pipe1.select_format()
        dirty1, emap1, _, _ = pipe1.corrupt(sample_df)

        pipe2 = CorruptionPipeline(seed=999, difficulty="medium")
        pipe2.select_format()
        dirty2, emap2, _, _ = pipe2.corrupt(sample_df)

        # At least the error maps should differ
        assert emap1 != emap2

    def test_easy_has_fewer_corruptions_than_hard(self, sample_df: pd.DataFrame):
        pipe_easy = CorruptionPipeline(seed=42, difficulty="easy")
        pipe_easy.select_format()
        _, _, _, meta_easy = pipe_easy.corrupt(sample_df)

        pipe_hard = CorruptionPipeline(seed=42, difficulty="hard")
        pipe_hard.select_format()
        _, _, _, meta_hard = pipe_hard.corrupt(sample_df)

        n_easy = len(meta_easy["corruptions_applied"])
        n_hard = len(meta_hard["corruptions_applied"])
        assert n_easy <= n_hard, (
            f"easy applied {n_easy} corruptions but hard only {n_hard}"
        )


class TestDifficultyProfiles:
    """Validate difficulty profiles and weights."""

    def test_difficulty_profiles_valid(self):
        for name, profile in DIFFICULTY_PROFILES.items():
            assert "num_corruption_types" in profile
            lo, hi = profile["num_corruption_types"]
            assert lo <= hi
            assert "fraction_range" in profile
            flo, fhi = profile["fraction_range"]
            assert 0 < flo < fhi <= 1.0
            assert "format_pool" in profile
            assert len(profile["format_pool"]) > 0

        assert set(DIFFICULTY_WEIGHTS.keys()) == set(DIFFICULTY_PROFILES.keys())
        assert abs(sum(DIFFICULTY_WEIGHTS.values()) - 1.0) < 1e-9


class TestRegistry:
    """Registry completeness and structure."""

    def test_all_corruption_functions_registered(self):
        assert len(CORRUPTION_REGISTRY) == 22, (
            f"Expected 22 corruption functions, got {len(CORRUPTION_REGISTRY)}"
        )

    def test_all_severity_entries_present(self):
        assert set(CORRUPTION_REGISTRY.keys()) == set(CORRUPTION_SEVERITY.keys())

    def test_registry_entries_have_required_keys(self):
        for name, meta in CORRUPTION_REGISTRY.items():
            assert "fn" in meta, f"{name} missing 'fn'"
            assert callable(meta["fn"]), f"{name} fn not callable"
            assert "requires_numeric" in meta, f"{name} missing 'requires_numeric'"
            assert "requires_string" in meta, f"{name} missing 'requires_string'"


class TestSelectFormatBeforeCorrupt:
    """RNG ordering: select_format() before corrupt() is critical."""

    def test_select_format_before_corrupt_reproducible(self, sample_df: pd.DataFrame):
        # Correct order: format then corrupt
        pipe1 = CorruptionPipeline(seed=77, difficulty="hard")
        fmt1 = pipe1.select_format()
        dirty1, emap1, _, _ = pipe1.corrupt(sample_df)

        pipe2 = CorruptionPipeline(seed=77, difficulty="hard")
        fmt2 = pipe2.select_format()
        dirty2, emap2, _, _ = pipe2.corrupt(sample_df)

        assert fmt1 == fmt2
        pd.testing.assert_frame_equal(dirty1, dirty2)
        assert emap1 == emap2


class TestErrorMapStructure:
    """Error map keys must be valid."""

    def _corrupt_with_only(
        self,
        sample_df: pd.DataFrame,
        corruptions: list[str],
    ) -> tuple[pd.DataFrame, dict, dict, dict]:
        pipe = CorruptionPipeline(seed=42, difficulty="medium")
        pipe.select_format()
        pipe.profile = {
            **pipe.profile,
            "allowed_corruptions": corruptions,
            "num_corruption_types": (len(corruptions), len(corruptions)),
            "fraction_range": (0.2, 0.2),
        }
        return pipe.corrupt(sample_df)

    def test_error_map_keys_valid(self, sample_df: pd.DataFrame):
        pipe = CorruptionPipeline(seed=42, difficulty="medium")
        pipe.select_format()
        _, error_map, _, _ = pipe.corrupt(sample_df)

        for key in error_map["cell_errors"]:
            parts = key.split(",", 1)
            assert len(parts) == 2, f"Invalid cell_error key: {key}"
            row_str, col = parts
            assert row_str.isdigit(), f"Row part not numeric: {key}"

        for key in error_map["spurious_rows"]:
            assert key.isdigit(), f"Invalid spurious_rows key: {key}"

        for key in error_map["missing_rows"]:
            assert key.isdigit(), f"Invalid missing_rows key: {key}"

        # Every cell_error should have required fields
        for key, info in error_map["cell_errors"].items():
            assert "severity" in info
            assert "clean_value" in info
            assert "corruption" in info
            assert isinstance(info["severity"], (int, float))
            assert info["corruption"] in CORRUPTION_SEVERITY

    def test_missing_rows_keys_strip_internal_prefix(self, sample_df: pd.DataFrame):
        _, error_map, _, _ = self._corrupt_with_only(sample_df, ["drop_rows"])

        assert error_map["missing_rows"]
        for key in error_map["missing_rows"]:
            assert key.isdigit()
            assert not key.startswith("missing_")

    def test_row_error_entries_include_corruption_type(self, sample_df: pd.DataFrame):
        _, error_map, _, _ = self._corrupt_with_only(
            sample_df,
            ["drop_rows", "duplicate_rows"],
        )

        row_errors = [
            *error_map["missing_rows"].values(),
            *error_map["spurious_rows"].values(),
        ]
        assert row_errors
        for info in row_errors:
            assert info["corruption"] in CORRUPTION_SEVERITY

    def test_severity_map_covers_every_error_map_entry(self, sample_df: pd.DataFrame):
        _, error_map, severity_map, _ = self._corrupt_with_only(
            sample_df,
            ["drop_rows", "duplicate_rows", "type_mangle"],
        )

        expected_total = sum(
            info["severity"]
            for group in error_map.values()
            for info in group.values()
        )

        assert severity_map["total_severity"] == pytest.approx(expected_total)
        assert sum(severity_map["by_type"].values()) == pytest.approx(expected_total)
        assert sum(severity_map["by_column"].values()) == pytest.approx(expected_total)
        for key in error_map["cell_errors"]:
            _, col = key.split(",", 1)
            assert col in severity_map["by_column"]
        if error_map["missing_rows"]:
            assert "__missing_rows__" in severity_map["by_column"]
        if error_map["spurious_rows"]:
            assert "__spurious_rows__" in severity_map["by_column"]


class TestPerformance:
    """Pipeline must handle larger datasets within time bounds."""

    def test_performance_5k_rows(self, large_df: pd.DataFrame):
        pipe = CorruptionPipeline(seed=42, difficulty="hard")
        pipe.select_format()

        start = time.perf_counter()
        dirty_df, error_map, severity_map, meta = pipe.corrupt(large_df)
        elapsed = time.perf_counter() - start

        assert elapsed < 2.0, f"Pipeline took {elapsed:.2f}s on 5K rows (limit: 2s)"
        assert len(dirty_df) > 0
        assert severity_map["total_severity"] > 0
