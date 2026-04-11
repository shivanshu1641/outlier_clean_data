# ADR 002: Per-dataset difficulty variants over combined hard task

## Status
Accepted (April 2026)

## Context
The original setup had three tasks: `easy_titanic`, `medium_wine`, `hard_combined`. The hard task was an artificial merge of Titanic + Wine columns, which created an unrealistic dataset with mixed domains and required pre-built task files.

## Decision
Each dataset supports three difficulty profiles:
- `easy`
- `medium`
- `hard`

Profiles live in `server/corruption/profiles.py` and are applied dynamically at `reset()` time, not generated into static task JSON files.

| difficulty | corruption fraction | corruption types |
|-----------|--------------------|--------------------|
| easy      | 5-10%             | 1-2 types (nulls, whitespace) |
| medium    | 12-15%            | 3-4 types (+ type_mangle, duplicates, outliers) |
| hard      | 30%               | all 5-6 types |

## Consequences
- Any catalog dataset can run at easy, medium, or hard
- Evaluation covers difficulty gradient within each dataset
- No artificial cross-domain datasets
- Difficulty changes do not require rebuilding task artifacts
