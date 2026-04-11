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

## Invariants (added post-audit)
- **Corruption ordering**: Row-level operations (`drop_rows`, `duplicate_rows`, `header_in_data`) are always applied before cell-level corruptions. Sort key: `(c not in ROW_CORRUPTIONS, c)` — row ops first, alphabetical within each group. This ensures cell error_map keys reference the final row structure after all index resets.
- **error_map key format**: `missing_rows` and `spurious_rows` both use bare digit strings (e.g., `"5"`, `"42"`) as dict keys — NOT prefixed. The grader expects this. The "missing_"/"spurious_" prefix is only used in the internal `error_log` list; it is stripped when building the error_map.
- **severity_map.by_type**: includes row-level corruptions (`drop_rows`, `duplicate_rows`, `header_in_data`) in addition to cell-level types.
