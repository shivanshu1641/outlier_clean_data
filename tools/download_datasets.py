"""Download datasets from catalog, validate, and save as clean CSVs.

Download order per dataset:
  1. GitHub mirror URLs — no auth needed, covers popular datasets
  2. Primary source_url from catalog — often UCI archive (can be unreliable)

Usage:
    python tools/download_datasets.py                 # download all
    python tools/download_datasets.py titanic iris    # download specific
    python tools/download_datasets.py --list          # list catalog entries
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import urllib.request
from io import StringIO
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CATALOG_PATH = ROOT / "datasets" / "catalog.json"
CLEAN_DIR = ROOT / "data" / "clean"

MAX_ROWS = 20_000
MAX_COLS = 30

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── GitHub mirror URLs ──────────────────────────────────────────────────
# catalog_name -> (url, csv_params_override | None)
# csv_params_override=None means use catalog csv_params as-is.

GITHUB_MIRRORS: dict[str, tuple[str, dict | None]] = {
    "iris": ("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv", None),
    "heart_disease": ("https://raw.githubusercontent.com/sharmaroshan/Heart-UCI-Dataset/master/heart.csv", None),
    "glass": ("https://raw.githubusercontent.com/jbrownlee/Datasets/master/glass.csv",
              {"header": None, "names": ["RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe", "Type"]}),
    "ionosphere": ("https://raw.githubusercontent.com/jbrownlee/Datasets/master/ionosphere.csv", {"header": None}),
    "sonar": ("https://raw.githubusercontent.com/jbrownlee/Datasets/master/sonar.csv", {"header": None}),
    "abalone": ("https://raw.githubusercontent.com/jbrownlee/Datasets/master/abalone.csv",
                {"header": None, "names": ["Sex", "Length", "Diameter", "Height", "Whole_weight",
                                           "Shucked_weight", "Viscera_weight", "Shell_weight", "Rings"]}),
    "breast_cancer": ("https://raw.githubusercontent.com/jbrownlee/Datasets/master/breast-cancer-wisconsin.csv",
                      {"header": None}),
    "auto_mpg": ("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv", None),
    "haberman": ("https://raw.githubusercontent.com/jbrownlee/Datasets/master/haberman.csv",
                 {"header": None, "names": ["age", "year", "nodes", "status"]}),
    "ecoli": ("https://raw.githubusercontent.com/jbrownlee/Datasets/master/ecoli.csv",
              {"header": None, "names": ["name", "mcg", "gvh", "lip", "chg", "aac", "alm1", "alm2", "class"]}),
    "german_credit": ("https://raw.githubusercontent.com/jbrownlee/Datasets/master/german.csv", {"header": None}),
    "wine_recognition": ("https://raw.githubusercontent.com/jbrownlee/Datasets/master/wine.csv", {"header": None}),
    "mushroom": ("https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/mushrooms.csv",
                 None),
    "seeds": ("https://raw.githubusercontent.com/selva86/datasets/master/seeds.csv", None),
    "wine_quality": ("https://raw.githubusercontent.com/dsrscientist/dataset1/master/winequality-red.csv",
                     {"sep": ";"}),
    "horse_colic": ("https://raw.githubusercontent.com/jbrownlee/Datasets/master/horse-colic.csv",
                    {"header": None, "sep": r"\s+"}),
    "thyroid_disease": ("https://raw.githubusercontent.com/jbrownlee/Datasets/master/new-thyroid.csv",
                        {"header": None}),
}


# ── helpers ──────────────────────────────────────────────────────────────


def load_catalog() -> dict:
    """Load and return the dataset catalog as a dict."""
    with open(CATALOG_PATH) as f:
        return json.load(f)


def _fetch_url(url: str, timeout: int = 120) -> str:
    """Download a URL and return its text content."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


def _read_csv(text: str, csv_params: dict | None) -> pd.DataFrame:
    """Parse CSV text into a DataFrame using optional csv_params."""
    params = dict(csv_params or {})
    params.pop("max_cols", None)
    return pd.read_csv(StringIO(text), **params)


def _validate_and_trim(
    df: pd.DataFrame, name: str, entry: dict
) -> pd.DataFrame | None:
    """Apply row/column caps and drop empty columns."""
    empty_cols = [c for c in df.columns if df[c].isna().all()]
    if empty_cols:
        log.info("  dropping %d empty column(s): %s", len(empty_cols), empty_cols[:5])
        df = df.drop(columns=empty_cols)

    row_cap = entry.get("max_rows", MAX_ROWS)
    if len(df) > row_cap:
        log.info("  trimming rows: %d -> %d", len(df), row_cap)
        df = df.head(row_cap)

    max_cols = (entry.get("csv_params") or {}).get("max_cols", MAX_COLS)
    if len(df.columns) > max_cols:
        log.info("  trimming cols: %d -> %d", len(df.columns), max_cols)
        df = df.iloc[:, :max_cols]

    if df.empty:
        log.warning("  %s: dataframe is empty after validation — skipping", name)
        return None

    return df


# ── download backends ────────────────────────────────────────────────────


def _try_github_mirror(name: str) -> pd.DataFrame | None:
    """Try downloading from a known GitHub mirror URL."""
    if name not in GITHUB_MIRRORS:
        return None

    url, csv_params = GITHUB_MIRRORS[name]
    log.info("  [github] trying %s ...", url)
    try:
        text = _fetch_url(url)
        return _read_csv(text, csv_params)
    except Exception as exc:
        log.warning("  [github] failed: %s", exc)
        return None


def _try_source_url(name: str, entry: dict) -> pd.DataFrame | None:
    """Try downloading from the catalog's primary source_url."""
    url = entry["source_url"]
    log.info("  [source] trying %s ...", url)
    try:
        text = _fetch_url(url)
        return _read_csv(text, entry.get("csv_params"))
    except Exception as exc:
        log.warning("  [source] failed: %s", exc)
        return None


# ── main pipeline ────────────────────────────────────────────────────────


def download_one(name: str, entry: dict, dest_dir: Path) -> bool:
    """Download, validate, and save a single dataset. Returns True on success."""
    filename = entry.get("filename", f"{name}.csv")
    out_path = dest_dir / filename

    if out_path.exists():
        log.info("  [skip] %s already exists", out_path.name)
        return True

    df = _try_github_mirror(name)
    if df is None:
        df = _try_source_url(name, entry)

    if df is None:
        log.error("  FAILED to download %s from all sources", name)
        return False

    df = _validate_and_trim(df, name, entry)
    if df is None:
        return False

    df.to_csv(out_path, index=False)
    log.info("  saved %s  (%d rows x %d cols)", filename, len(df), len(df.columns))
    return True


def download_all(
    names: list[str] | None = None,
    catalog: dict | None = None,
    dest_dir: Path | None = None,
) -> dict[str, bool]:
    """Download datasets from the catalog.

    Args:
        names: Specific dataset names to download (None = all).
        catalog: Pre-loaded catalog dict (loads from disk if None).
        dest_dir: Output directory (defaults to data/clean).

    Returns:
        Dict mapping dataset name -> success bool.
    """
    if catalog is None:
        catalog = load_catalog()
    if dest_dir is None:
        dest_dir = CLEAN_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)

    targets = names if names else list(catalog.keys())
    results: dict[str, bool] = {}

    for name in targets:
        if name not in catalog:
            log.warning("'%s' not found in catalog — skipping", name)
            results[name] = False
            continue

        log.info("[%s]", name)
        ok = download_one(name, catalog[name], dest_dir)
        results[name] = ok
        time.sleep(0.25)

    succeeded = sum(v for v in results.values())
    log.info(
        "done: %d/%d succeeded, %d failed",
        succeeded,
        len(results),
        len(results) - succeeded,
    )
    return results


# ── CLI ──────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Download datasets from catalog")
    parser.add_argument(
        "names",
        nargs="*",
        help="Dataset names to download (default: all)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all catalog entries and exit",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=CLEAN_DIR,
        help="Output directory (default: data/clean)",
    )
    args = parser.parse_args()

    catalog = load_catalog()

    if args.list:
        for name, entry in sorted(catalog.items()):
            print(
                f"  {name:30s}  {entry['size_class']:6s}  "
                f"{entry['rows']:>6d} rows  {entry['cols']:>3d} cols  "
                f"[{entry['domain']}]"
            )
        print(f"\n  Total: {len(catalog)} datasets")
        sys.exit(0)

    results = download_all(
        names=args.names or None,
        catalog=catalog,
        dest_dir=args.dest,
    )
    failed = [k for k, v in results.items() if not v]
    if failed:
        log.warning("failed to download %d datasets: %s", len(failed), ", ".join(failed))


if __name__ == "__main__":
    main()
