# Runbook: Prepare Datasets

## When to run
- First-time setup
- After adding datasets to `catalog.json`
- When the local dataset cache needs to be refreshed

## Command
```bash
source .venv/bin/activate
python tools/download_datasets.py              # download all 25 datasets
python tools/download_datasets.py titanic iris  # download specific ones
python tools/download_datasets.py --list        # list catalog entries
```

## What it does
Downloads clean CSVs from `catalog.json` (25 datasets) into `data/clean/`. The download script tries GitHub mirror URLs first, then falls back to the catalog's primary `source_url`.

The environment does NOT need pre-built dirty CSVs or task files — corruption happens dynamically at `reset()` time.

## Download sources
1. **GitHub mirrors** — verified working URLs for popular datasets (no auth needed)
2. **Primary source_url** — from catalog, often UCI archive (can be unreliable)

## Format generation
Non-CSV formats (JSON, Excel, XML, TSV, etc.) are generated at corruption time by `CorruptionPipeline.select_format()`. Only clean CSVs need to be downloaded.

## Notes
- Catalog was trimmed from 118 to 25 datasets (only reliably downloadable ones retained)
- Some datasets have auto-inferred semantic rules stored in their catalog entries
- Seeded resets produce reproducible dirty datasets
- Add or update dataset entries in `catalog.json`, then rerun the download command
