# Architecture

## Overview

A generative data cleaning RL environment where an AI agent receives dirty tabular data in one of several file formats and must fix it using Python data-cleaning code, evaluated against the clean ground truth.

There are no pre-built dirty files. Each `reset()` call runs the `CorruptionPipeline` fresh against a clean dataset, producing dirty content, an error map, format metadata, and hints.

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
    → CorruptionPipeline(seed, difficulty, category).select_format()   ← must precede corrupt()
    → CorruptionPipeline.corrupt(clean_df, rules=rules)
        ├─ value corruptions       (nulls, typos, type mangling, outliers, swaps, whitespace, …)
        ├─ row/schema corruptions  (drop rows, duplicate rows, header injection)
        └─ format-specific corruptions  (delimiter noise, encoding shifts, nesting errors, …)
        → produces: dirty_df, error_map, severity_map, metadata, hints
    → create sandbox:
        write input.<format>        (original dirty raw file, never modified)
        write input.csv             (normalized dirty CSV)
        write current.csv           (working copy)
        write checkpoints/step_000.csv  (initial dirty state for undo)
    → grade(clean_df, current_df, error_map, metadata, rules=rules) → initial reward
    → return Observation(file_format, target_schema, file_preview, diagnosis, semantic_rules, hints)

  POST /step  {type: "explore", query: "df.head()"}
       ↓
  execute_explore(query, sandbox_dir)  ← persistent worker, read-only eval mode
    → return Observation (no state change, small efficiency cost)

  POST /step  {type: "transform", code: "df['Age'].fillna(...)"}
       ↓
  save checkpoint (checkpoints/step_NNN.csv)
    → execute_transform(code, sandbox_dir) ← persistent worker, AST-checked exec mode
    → overwrite current.csv
    → grade(clean_df, result_df, error_map, metadata) → updated reward
    → return Observation

  POST /step  {type: "undo"}
       ↓
  restore previous checkpoint
    → send "reload" to worker (resyncs in-memory df from current.csv)
    → increment undo_count
    → re-grade current.csv with undo cost
    → return Observation

  POST /step  {type: "validate"}
       ↓
  if validate budget remains (2 per episode):
    → grade + build structured per-column / per-corruption-type diagnosis
    → increment validate_count / validate_uses
    → return Observation(validate_result=structured_diagnosis)

  POST /step  {type: "done"}
       ↓
  if first done AND score < 1.0:
    → soft done: return score + remaining errors (done=False, agent continues)
  else:
    → hard done: final grade, return Observation(done=True)

[Client]
  DataCleaningClient(base_url=ENV_URL)
    .reset(task_id=...) → StepResult
    .step(action)       → StepResult

[Inference]
  llm_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
      ↑ validator/deployed runs must use injected proxy env vars directly
  for task in eval_task_ids:   # 25 pinned (dataset, difficulty, format) combos
    obs = env.reset(task_id)
    while not done:
      messages = [system_prompt, user_prompt(obs)]
      action = llm_client.chat.completions.create(...)
      obs = env.step(action)   # explore / transform / undo / validate / done
```

## Component Responsibilities

| Component | Owns |
|-----------|------|
| `datasets/` | Dataset catalog and materialized inputs for generated episodes |
| `tools/download_datasets.py` | Download/materialization pipeline for the dataset catalog |
| `server/corruption/pipeline.py` | `CorruptionPipeline`, format selection, dynamic corruption orchestration, error_map/severity_map generation |
| `server/corruption/value_corruptions.py` | 22 value-level corruptions: nulls, typos, type mangling, outliers, swaps, formatting noise, decimal shifts, whitespace, value swaps |
| `server/corruption/format_corruptions.py` | 9 output formats (csv, json, jsonl, excel, tsv, xml, fixed_width, html_table, sql_dump) and ~40 format-specific corruptions |
| `server/corruption/hints.py` | 3 hint levels: strategy, tactical, categorical |
| `server/corruption/profiles.py` | Easy/medium/hard difficulty profiles |
| `server/corruption/categories.py` | 6 benchmark categories (FP/VR/MD/SR/SV/CP) mapping to corruption subsets and format pools |
| `server/rules/` | 7 semantic rule types, auto-inference from clean data, grading validation, catalog enrichment |
| `server/grader.py` | Multi-level scoring, content-based row matching with numeric normalization, collateral damage via row_mapping, validation diagnostics, action costs, semantic scoring |
| `server/environment.py` | Episode lifecycle, generative reset, `LEGACY_TASK_MAP`, action dispatch, checkpoint/undo coordination, observation building, soft-done logic |
| `server/sandbox.py` | AST safety (exec + eval modes), persistent worker management, raw-format file setup, CSV working copy, checkpoint save/restore, env stripping, atexit cleanup |
| `server/worker.py` | Agent code execution with restricted `__builtins__`, pandas/numpy + io/openpyxl/yaml/lxml namespace, inplace rewriting, `_BoundedStringIO` stdout cap, 2GB memory limit, reload-on-undo |
| `inference.py` | 25-task eval suite, LLM orchestration, prompt building, action parsing, auto-undo on regression, NaN coerce warnings |
| `tools/benchmark_runner.py` | CLI benchmark orchestrator: task matrix, inference runs, JSONL + CSV result persistence |
| `ui/app.py` | Gradio dashboard entry point with leaderboard, episode explorer, and catalog tabs |
| `ui/data_loader.py` | Loads benchmark results (JSONL), episode logs, and catalog.json for the UI |

## Grading

The grader receives:
- `clean_df` — ground truth
- `result_df` — agent's current DataFrame
- `error_map` — generated map of corrupted cells/rows/schema issues
- `metadata` / action counts — format/profile context plus transform, explore, undo, and validate counts
- `rules` (optional) — auto-inferred semantic constraints

The grader never knows about corruption implementations — it scores result quality from the above inputs alone.

### Reward formula

```
# With semantic rules (auto-inferred from clean data):
base_score =
  schema_score       × 0.15   # column names, count, structural compatibility
  row_score          × 0.15   # duplicate/missing/reordered row recovery (content-based matching)
  cell_score         × 0.50   # per-cell accuracy (see below)
  distribution_score × 0.10   # coarse statistical similarity (mean, std, quantiles)
  semantic_score     × 0.10   # rule violation rate against 7 constraint types

# Without rules:
base_score =
  schema_score       × 0.15
  row_score          × 0.20
  cell_score         × 0.55
  distribution_score × 0.10

efficiency_factor = max(0.5, 1.0 - transform_penalty - explore_penalty - undo_penalty - validate_penalty)
reward = base_score × efficiency_factor   [clamped to 0.0–1.0]
```

### Cell score mechanics

| Outcome | Severity |
|---------|----------|
| Fixed (result == clean value) | 0 remaining |
| Accepted null fill (within 0.5σ, matching accepted strategy) | 0 remaining |
| Numeric near-miss (≤5% relative error) | graduated partial credit |
| Unfixed (still dirty) | full remaining |
| Wrong value (changed to incorrect value) | **1.5× severity** |
| Collateral damage (correct cell agent broke) | **+0.5 severity per cell** |

### Row matching

Content-based matching via `_row_hash`. Handles reordered, dropped, and duplicated rows correctly without phantom collateral damage. `_row_hash` normalizes numeric values (`6.0 → "6"`) to survive CSV round-trips where int columns become float when NaN is present.

### Efficiency costs

| Action | Cost |
|--------|------|
| Successful explore | 0.01 efficiency |
| Timed-out explore | 0.03 efficiency |
| Undo | configured `undo_cost` |
| Validate | configured `validate_cost` |
| Transforms beyond minimum | proportional penalty |

The 0.5 floor ensures even heavy explorers get at least half credit for good cleaning.

### Semantic rule types

7 constraint types auto-inferred from each clean dataset and stored in `catalog.json`:

| Type | Checks |
|------|--------|
| Range | Numeric values within expected min/max |
| Regex | String values matching expected pattern |
| Enum | Categorical values within known set |
| Dtype | Column dtype matches expected type |
| NotNull | Non-nullable columns have no missing values |
| Unique | Unique-constraint columns have no duplicates |
| CrossColumn | Inter-column relationships |

## Sandbox Isolation

Each episode gets an isolated working directory:

```
outputs/sandbox/{episode_id}/
├── input.<format>     (original dirty raw file, never modified)
├── input.csv          (normalized dirty CSV)
├── current.csv        (working copy, updated after each transform)
├── checkpoints/       (step_000.csv = dirty state; step_NNN.csv after each transform)
├── scripts/           (generated transform scripts for replay)
└── artifacts/         (execution outputs and snapshots)
```

Agent code runs in a persistent subprocess per episode:

| Mechanism | Detail |
|-----------|--------|
| **AST safety scan** | Two modes: `exec` for transforms, `eval` for explores. Blocks `BLOCKED_MODULES` (os, sys, subprocess, socket, …), `BLOCKED_NAMES` (open, exec, eval, getattr, globals, locals, …), `BLOCKED_DUNDERS` (__class__, __subclasses__, __getattribute__, …) |
| **Restricted builtins** | Worker exec/eval namespace uses an explicit allowlist (~30 safe functions); `open`, `exec`, `compile`, `object`, `issubclass`, `getattr`, `globals` absent |
| **Stripped environment** | Worker inherits only `PATH`, `HOME`, `LANG`, `LC_ALL`, `PYTHONPATH`, `VIRTUAL_ENV`; server secrets never forwarded |
| **Process group isolation** | `start_new_session=True`; timeout kills the full process group |
| **Transform timeout** | 30 seconds |
| **Explore timeout** | 10 seconds |
| **Memory limit** | 2GB via `RLIMIT_AS` (Linux only) |
| **Stdout cap** | 1MB via `_BoundedStringIO` to prevent print-bomb OOM |
| **Persistent worker** | pandas/numpy loaded once per episode; transforms use `df.copy()` in exec namespace; explores use in-memory `df` directly (read-only) |
| **Undo sync** | Checkpoint restore sends `"reload"` to worker to resync in-memory df from `current.csv` |
| **Auto-rewrite inplace** | Worker converts `df['col'].fillna(val, inplace=True)` → `df['col'] = df['col'].fillna(val)` (pandas 2.x broke chained inplace) |
| **atexit cleanup** | `_active_workers` set + `atexit.register(_cleanup_all_workers)` in sandbox.py; `tini` in Dockerfile reaps zombies |
