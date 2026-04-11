# ADR 003: Generative corruption at reset() time

## Status
Accepted

## Context
v1 used pre-built dirty CSV files and static error maps stored in tasks/. This required running an offline generation script before running the environment, made task variety fixed, and didn't support seeded reproducibility across the full pipeline.

## Decision
Move all corruption to reset() time using a seeded CorruptionPipeline. The pipeline:
1. Selects a format via select_format(rng) — must be called before corrupt()
2. Applies 22 value-level corruptions to the DataFrame
3. Converts to the target format and applies format-specific corruptions
4. Returns (dirty_df, error_map, severity_map, metadata)

Seeded RNG ensures full reproducibility: same seed → identical dirty dataset.

## Consequences
- No offline generation step needed — any dataset from catalog.json works immediately
- 25 datasets × 3 difficulties × 9 formats × 6 categories = large effective task space
- LEGACY_TASK_MAP maps old task_id strings to dataset_id + difficulty for backward compat
- CorruptionPipeline.select_format() must always be called before corrupt() (order matters for RNG state)
- Catalog trimmed from 118 to 25 entries (only datasets with reliable download sources retained)
- 7 semantic rule types auto-inferred from clean data and stored in catalog entries
- 6 benchmark categories (FP/VR/MD/SR/SV/CP) constrain corruption selection and format pools
