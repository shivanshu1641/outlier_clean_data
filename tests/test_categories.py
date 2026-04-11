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
        registry_names = set(CORRUPTION_REGISTRY.keys())
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
        assert corruptions is None

    def test_get_formats_for_category(self):
        formats = get_formats_for_category("FP")
        assert "csv" not in formats
        assert "json" in formats

    def test_get_formats_for_vr_is_csv(self):
        formats = get_formats_for_category("VR")
        assert formats == ["csv"]

    def test_invalid_category_raises(self):
        with pytest.raises(ValueError):
            get_corruptions_for_category("INVALID")


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
        "active": rng.choice(["yes", "no"], size=n),
        "email": [f"person{i}@example.com" for i in range(n)],
    })


class TestCategoryAwarePipeline:
    def test_vr_category_only_uses_vr_corruptions(self, sample_df):
        pipe = CorruptionPipeline(seed=42, difficulty="medium", category="VR")
        pipe.select_format()
        dirty_df, error_map, sev, meta = pipe.corrupt(sample_df)
        applied = [c["type"] for c in meta.get("corruptions_applied", [])]
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
        applied = [c["type"] for c in meta.get("corruptions_applied", [])]
        md_set = set(CATEGORY_CORRUPTION_MAP["MD"])
        for name in applied:
            assert name in md_set

    def test_no_category_behaves_as_before(self, sample_df):
        pipe = CorruptionPipeline(seed=42, difficulty="medium")
        pipe.select_format()
        dirty_df, error_map, sev, meta = pipe.corrupt(sample_df)
        assert isinstance(dirty_df, pd.DataFrame)
        assert "corruptions_applied" in meta
