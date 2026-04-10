"""Download datasets from catalog, validate, and save as clean CSVs.

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
CLEAN_DIR = ROOT / "datasets" / "clean"

MAX_ROWS = 20_000
MAX_COLS = 30

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


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
    # try utf-8, fall back to latin-1
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


def _read_csv(text: str, csv_params: dict | None) -> pd.DataFrame:
    """Parse CSV text into a DataFrame using optional csv_params."""
    params = dict(csv_params or {})
    # Handle max_cols — not a pandas param, pop it out
    params.pop("max_cols", None)
    return pd.read_csv(StringIO(text), **params)


def _validate_and_trim(
    df: pd.DataFrame, name: str, entry: dict
) -> pd.DataFrame | None:
    """Apply row/column caps and drop empty columns.

    Returns the cleaned DataFrame or None if validation fails.
    """
    # drop fully-empty columns
    empty_cols = [c for c in df.columns if df[c].isna().all()]
    if empty_cols:
        log.info("  dropping %d empty column(s): %s", len(empty_cols), empty_cols[:5])
        df = df.drop(columns=empty_cols)

    # enforce per-dataset or global row cap
    row_cap = entry.get("max_rows", MAX_ROWS)
    if len(df) > row_cap:
        log.info("  trimming rows: %d -> %d", len(df), row_cap)
        df = df.head(row_cap)

    # enforce column cap (take first MAX_COLS columns)
    max_cols = (entry.get("csv_params") or {}).get("max_cols", MAX_COLS)
    if len(df.columns) > max_cols:
        log.info("  trimming cols: %d -> %d", len(df.columns), max_cols)
        df = df.iloc[:, :max_cols]

    if df.empty:
        log.warning("  %s: dataframe is empty after validation — skipping", name)
        return None

    return df


# ── main pipeline ────────────────────────────────────────────────────────


def download_one(name: str, entry: dict, dest_dir: Path) -> bool:
    """Download, validate, and save a single dataset. Returns True on success."""
    url = entry["source_url"]
    filename = entry.get("filename", f"{name}.csv")
    out_path = dest_dir / filename

    if out_path.exists():
        log.info("  [skip] %s already exists", out_path.name)
        return True

    log.info("  downloading %s ...", url)
    try:
        text = _fetch_url(url)
    except Exception as exc:
        log.error("  FAILED to download %s: %s", name, exc)
        return False

    try:
        df = _read_csv(text, entry.get("csv_params"))
    except Exception as exc:
        log.error("  FAILED to parse %s: %s", name, exc)
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
        dest_dir: Output directory (defaults to datasets/clean).

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
        # small politeness delay between downloads
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
        help="Output directory (default: datasets/clean)",
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
    # exit 1 if any failed
    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
