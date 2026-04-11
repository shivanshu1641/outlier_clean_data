# Runbook: Prepare Datasets

## When to run
- First-time setup
- After adding datasets to `catalog.json`
- When the local dataset cache needs to be refreshed

## Command
```bash
source .venv/bin/activate
python tools/download_datasets.py
```

## What it does
Datasets are defined in `catalog.json` and downloaded into the local data cache. The environment no longer needs pre-built dirty CSVs, error maps, severity maps, or task JSON files.

## v2 flow
Corruption happens dynamically at `reset()` time using the selected dataset, difficulty, format, and seed. Any downloaded dataset from `catalog.json` can be used immediately.

## Notes
- No offline corruption generation step is needed
- Seeded resets produce reproducible dirty datasets
- Add or update dataset entries in `catalog.json`, then rerun the download command
