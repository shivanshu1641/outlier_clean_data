"""Tests for the dataset catalog and download pipeline."""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
CATALOG_PATH = ROOT / "datasets" / "catalog.json"

REQUIRED_DATASETS = [
    "titanic",
    "adult_income",
    "heart_disease",
    "iris",
    "wine_quality",
    "auto_mpg",
    "abalone",
    "mushroom",
    "breast_cancer",
]

VALID_SIZE_CLASSES = {"small", "medium", "large"}

REQUIRED_FIELDS = {"source_url", "filename", "domain", "rows", "cols", "dtypes", "size_class"}

REQUIRED_DTYPE_KEYS = {"numeric", "categorical", "datetime", "text"}


# ── Catalog file tests ───────────────────────────────────────────────────


def test_catalog_file_exists():
    assert CATALOG_PATH.exists(), f"catalog.json not found at {CATALOG_PATH}"


def test_catalog_valid_json():
    text = CATALOG_PATH.read_text()
    catalog = json.loads(text)
    assert isinstance(catalog, dict), "catalog must be a JSON object"


@pytest.fixture
def catalog():
    return json.loads(CATALOG_PATH.read_text())


def test_catalog_has_at_least_15_entries(catalog):
    assert len(catalog) >= 15, f"Catalog has only {len(catalog)} entries, need >= 15"


def test_catalog_has_100_plus_entries(catalog):
    assert len(catalog) >= 100, f"Catalog has only {len(catalog)} entries, need >= 100"


def test_required_datasets_present(catalog):
    missing = [ds for ds in REQUIRED_DATASETS if ds not in catalog]
    assert not missing, f"Missing required datasets: {missing}"


# ── Per-entry structure tests ────────────────────────────────────────────


def test_all_entries_have_required_fields(catalog):
    for name, entry in catalog.items():
        missing = REQUIRED_FIELDS - set(entry.keys())
        assert not missing, f"'{name}' missing fields: {missing}"


def test_all_entries_have_valid_size_class(catalog):
    for name, entry in catalog.items():
        assert entry["size_class"] in VALID_SIZE_CLASSES, (
            f"'{name}' has invalid size_class: {entry['size_class']}"
        )


def test_all_entries_have_valid_urls(catalog):
    for name, entry in catalog.items():
        url = entry["source_url"]
        assert url.startswith("http"), (
            f"'{name}' has invalid URL (must start with http): {url}"
        )


def test_all_filenames_unique(catalog):
    filenames = [entry["filename"] for entry in catalog.values()]
    dupes = [f for f in filenames if filenames.count(f) > 1]
    assert not dupes, f"Duplicate filenames: {set(dupes)}"


def test_all_filenames_end_with_csv(catalog):
    for name, entry in catalog.items():
        assert entry["filename"].endswith(".csv"), (
            f"'{name}' filename does not end with .csv: {entry['filename']}"
        )


def test_dtypes_have_required_keys(catalog):
    for name, entry in catalog.items():
        dt = entry["dtypes"]
        missing = REQUIRED_DTYPE_KEYS - set(dt.keys())
        assert not missing, f"'{name}' dtypes missing keys: {missing}"


def test_rows_and_cols_positive(catalog):
    for name, entry in catalog.items():
        assert entry["rows"] > 0, f"'{name}' rows must be > 0"
        assert entry["cols"] > 0, f"'{name}' cols must be > 0"


def test_size_class_consistent_with_rows(catalog):
    """size_class should roughly match the row count."""
    for name, entry in catalog.items():
        rows = entry["rows"]
        sc = entry["size_class"]
        if sc == "small":
            assert rows < 1000, f"'{name}' is 'small' but has {rows} rows"
        elif sc == "medium":
            assert 1000 <= rows <= 10000, (
                f"'{name}' is 'medium' but has {rows} rows"
            )
        elif sc == "large":
            assert rows > 10000, f"'{name}' is 'large' but has {rows} rows"


# ── Domain diversity ─────────────────────────────────────────────────────


def test_at_least_5_distinct_domains(catalog):
    domains = {entry["domain"] for entry in catalog.values()}
    assert len(domains) >= 5, f"Only {len(domains)} domains, need >= 5"


# ── Download pipeline import test ────────────────────────────────────────


def test_download_module_importable():
    """Ensure the download module can be imported without errors."""
    import importlib
    mod = importlib.import_module("tools.download_datasets")
    assert hasattr(mod, "load_catalog")
    assert hasattr(mod, "download_all")
    assert hasattr(mod, "download_one")
