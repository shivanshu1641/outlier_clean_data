# Architecture

## Overview

A generative data cleaning RL environment where an AI agent receives dirty tabular data in one of several file formats and must fix it using Python data-cleaning code, evaluated against the clean ground truth.

## Data Flow

```
[Datasets]
  datasets/catalog.json + materialized dataset files
       ↓
[Server]
  POST /reset?task_id=<eval_or_legacy_id>&category=<FP|VR|MD|SR|SV|CP>
       ↓
  DataCleaningEnvironment.reset()
    → resolve task/profile via EVAL_TASK_IDS or LEGACY_TASK_MAP
    → load source dataset + semantic rules from catalog
    → CorruptionPipeline(seed, difficulty, category).select_format()
    → CorruptionPipeline.corrupt(clean_df, rules=rules)
        - value corruptions
        - row/schema corruptions
        - format-specific corruptions
        - hints + diagnosis metadata
    → create sandbox with raw dirty file + normalized current.csv
    → save initial checkpoint
    → grade(clean_df, current_df, error_map, metadata, rules=rules) → initial reward
    → return Observation(file_format, target_schema, file_preview, diagnosis, semantic_rules)

  POST /step  {type: "explore", query: "df.head()"}
       ↓
  execute_explore(query, sandbox_dir)  ← persistent worker, read-only
    → return Observation (no state change)

  POST /step  {type: "transform", code: "df['Age'].fillna(...)"}
       ↓
  save checkpoint
    → execute_transform(code, sandbox_dir) ← persistent worker, AST-checked
    → overwrite current.csv
    → grade(clean_df, result_df, error_map, metadata) → updated reward
    → return Observation

  POST /step  {type: "undo"}
       ↓
  restore previous checkpoint
    → increment undo_count
    → re-grade current.csv with undo cost
    → return Observation

  POST /step  {type: "validate"}
       ↓
  if validate budget remains:
    → grade + build validation diagnosis
    → increment validate_count / validate_uses
    → return Observation(validate_result=...)

  POST /step  {type: "done"}
       ↓
  final grade → return Observation (done=True)

[Client]
  DataCleaningClient(base_url=ENV_URL)
    .reset(task_id=...) → StepResult
    .step(action)       → StepResult

[Inference]
  llm_client = OpenAI(base_url=os.environ["API_BASE_URL"], api_key=os.environ["API_KEY"])
      ↑ validator/deployed runs must use injected proxy env vars directly
  for task in task_ids:
    obs = env.reset(task_id)
    while not done:
      messages = [system_prompt, user_prompt(obs)]
      action = llm_client.chat.completions.create(...)
      obs = env.step(action)  ← explore/transform/undo/validate/done
```

## Component Responsibilities

| Component | Owns |
|-----------|------|
| `datasets/` | Dataset catalog and materialized inputs for generated episodes |
| `tools/download_datasets.py` | Download/materialization pipeline for the dataset catalog |
| `server/corruption/pipeline.py` | `CorruptionPipeline`, format selection, dynamic corruption orchestration |
| `server/corruption/value_corruptions.py` | 22 value-level corruptions such as nulls, typos, type issues, outliers, swaps, and formatting noise |
| `server/corruption/format_corruptions.py` | Supported output formats (csv, json, jsonl, excel, tsv, xml, fixed_width, html_table, sql_dump, yaml) and ~40 format-specific corruptions |
| `server/corruption/hints.py` | Three hint levels: strategy, tactical, categorical |
| `server/corruption/profiles.py` | Easy/medium/hard difficulty profiles |
| `server/rules/` | 7 semantic rule types, auto-inference from clean data, validation, catalog enrichment |
| `server/corruption/categories.py` | 6 benchmark categories (FP/VR/MD/SR/SV/CP) mapping to corruption subsets and format pools |
| `server/grader.py` | Multi-level scoring, content-based row matching, validation diagnostics, action costs, semantic scoring |
| `server/environment.py` | Episode lifecycle, dynamic reset, action dispatch, checkpoint/undo coordination, observation building |
| `server/sandbox.py` | Code safety, persistent worker management, raw-format file setup, CSV working copy, checkpoint save/restore |
| `server/worker.py` | Agent code execution with pandas/numpy plus `io`, `openpyxl`, `yaml`, and `lxml` in namespace |
| `inference.py` | 15-task eval suite, LLM orchestration, prompt building, action parsing, undo/validate support |
| `tools/benchmark_runner.py` | CLI benchmark orchestrator — generates task matrix, runs inference agent, saves JSONL + CSV results |
| `ui/app.py` | Gradio dashboard entry point with leaderboard, episode explorer, and catalog tabs |
| `ui/data_loader.py` | Loads benchmark results (JSONL), episode logs, and catalog.json for the UI |

## Grading

The grader receives:
- `clean_df` — ground truth
- `result_df` — agent's current DataFrame
- `error_map` — generated map of corrupted cells/rows/schema issues
- `metadata` / action counts — format/profile context plus transform, explore, undo, and validate counts

The reward combines up to five quality levels (semantic only when rules present):
- **schema_score (0.15)** — expected columns, names, and structural compatibility
- **row_score (0.15/0.20)** — duplicate/missing/reordered row recovery using content-based matching
- **cell_score (0.50/0.55)** — exact cell fixes, accepted null fills, numeric near-misses, wrong-value penalties, whitespace tolerance, and collateral damage
- **distribution_score (0.10)** — coarse statistical similarity to the clean data
- **semantic_score (0.10)** — rule violation rate against auto-inferred or manual constraints (only when rules present)

```
# With semantic rules:
base_score =
  schema_score       × 0.15 +
  row_score          × 0.15 +
  cell_score         × 0.50 +
  distribution_score × 0.10 +
  semantic_score     × 0.10

# Without rules (legacy):
base_score =
  schema_score       × 0.15 +
  row_score          × 0.20 +
  cell_score         × 0.55 +
  distribution_score × 0.10

efficiency_factor = max(0.5, 1 - transform_penalty - explore_penalty - undo_penalty - validate_penalty)
reward = base_score × efficiency_factor
```

`validate` actions return diagnostic feedback and are limited to 2 uses per episode. `undo` restores the latest checkpoint and is included in the action-cost accounting.

## Sandbox Isolation

Each episode gets an isolated directory:
```
outputs/sandbox/{episode_id}/
├── input.<format>     (original dirty raw file, never modified)
├── input.csv          (normalized dirty CSV)
├── current.csv        (working copy, updated after each transform)
├── checkpoints/       (filesystem snapshots for undo)
├── scripts/           (generated transform scripts for replay)
└── artifacts/         (execution outputs/snapshots)
```

Agent code runs in a subprocess with:
- AST-level import/call blocking (no os, sys, subprocess, open, eval, etc.)
- 30s transform timeout, 10s explore timeout
- 2GB memory limit
- Persistent worker process per episode so pandas/numpy and supported parsing libraries stay warm
