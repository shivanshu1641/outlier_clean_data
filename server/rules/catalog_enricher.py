"""Enrich catalog.json with auto-inferred semantic rules.

Usage:
    python -m server.rules.catalog_enricher datasets/catalog.json data/clean/
"""
from __future__ import annotations

import json
import logging
import os
import sys

import pandas as pd

try:
    from server.rules.inferrer import infer_rules
    from server.rules.types import rule_to_dict
except ImportError:
    from rules.inferrer import infer_rules
    from rules.types import rule_to_dict

logger = logging.getLogger(__name__)


def enrich_catalog(catalog_path: str, clean_dir: str) -> None:
    """Read catalog (dict-shaped), infer rules per dataset, merge overrides, write back."""
    with open(catalog_path) as f:
        catalog = json.load(f)

    enriched_count = 0
    for dataset_id, entry in catalog.items():
        filename = entry.get("filename", "")
        csv_path = os.path.join(clean_dir, filename)
        if not os.path.exists(csv_path):
            logger.warning("Skipping %s: no clean file at %s", dataset_id, csv_path)
            continue

        try:
            csv_params = entry.get("csv_params", {})
            max_rows = entry.get("max_rows")
            df = pd.read_csv(csv_path, nrows=max_rows, **csv_params)
        except Exception as exc:
            logger.warning("Skipping %s: failed to read CSV: %s", dataset_id, exc)
            continue

        domain = entry.get("domain")
        inferred = infer_rules(df, domain=domain)
        inferred_dicts = [rule_to_dict(r) for r in inferred]

        overrides = entry.get("rules_override", [])
        merged = _merge_rules(inferred_dicts, overrides)

        entry["rules"] = merged
        enriched_count += 1

    with open(catalog_path, "w") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)

    logger.info("Enriched %d datasets in %s", enriched_count, catalog_path)


def _merge_rules(inferred: list[dict], overrides: list[dict]) -> list[dict]:
    """Merge inferred rules with manual overrides. Manual wins for same column+type."""

    def _key(r: dict) -> tuple:
        if "column" in r:
            return (r["type"], r["column"])
        if "columns" in r:
            return (r["type"], tuple(r["columns"]))
        return (r["type"], "")

    override_keys = {_key(r) for r in overrides}
    result = [r for r in inferred if _key(r) not in override_keys]
    result.extend(overrides)
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) < 3:
        print(f"Usage: python -m server.rules.catalog_enricher <catalog.json> <clean_dir>")
        sys.exit(1)
    enrich_catalog(sys.argv[1], sys.argv[2])
