# ADR 002: Per-dataset difficulty variants over combined hard task

## Status
Accepted (April 2026)

## Context
The original setup had three tasks: `easy_titanic`, `medium_wine`, `hard_combined`. The hard task was an artificial merge of Titanic + Wine columns, which created an unrealistic dataset with mixed domains.

## Decision
Each dataset gets three difficulty variants instead of one:
- `titanic_easy`, `titanic_medium`, `titanic_hard`
- `wine_easy`, `wine_medium`, `wine_hard`

Difficulty is controlled by corruption fractions and which corruption types are applied:

| difficulty | corruption fraction | corruption types |
|-----------|--------------------|--------------------|
| easy      | 5-10%             | 1-2 types (nulls, whitespace) |
| medium    | 12-15%            | 3-4 types (+ type_mangle, duplicates, outliers) |
| hard      | 30%               | all 5-6 types |

## Consequences
- 6 tasks total vs 3 before
- Evaluation covers difficulty gradient within each domain
- No artificial cross-domain datasets
- Dropped `hard_combined` — it was not a realistic cleaning scenario
