# Architecture

## Overview

A generative data cleaning RL environment. Each episode is created dynamically at `reset()` time: the corruption pipeline applies a configurable mix of value, row/schema, and format-specific corruptions to a clean dataset, then the grader scores the agent's repair code against the original ground truth using multi-level data quality metrics.

There are no pre-built dirty files. Everything is generated at runtime.

## Data Flow

```
[Datasets]
  datasets/catalog.json + materialized dataset files
       ↓
[Server: POST /reset]
  DataCleaningEnvironment.reset()
    → resolve task → load clean dataset + semantic rules from catalog
    → CorruptionPipeline(seed, difficulty).select_format()   ← must precede corrupt()
    → CorruptionPipeline.corrupt(clean_df, rules)
        ├─ value corruptions       (nulls, typos, type mangling, outliers, swaps, whitespace, …)
        ├─ row/schema corruptions  (drop rows, duplicate rows, header injection)
        └─ format-specific corruptions  (delimiter noise, encoding shifts, nesting errors, …)
    → create sandbox:
        write input.<format>  (original dirty raw file, never modified)
        write input.csv       (normalized dirty CSV)
        write current.csv     (working copy)
        write checkpoints/step_000.csv  (initial dirty state for undo)
    → grade(clean_df, current_df, error_map, metadata, rules) → initial reward
    → return Observation(file_format, file_preview, diagnosis, semantic_rules, hints)

[Server: POST /step]

  {type: "explore", query: "df.isnull().sum()"}
    → persistent worker (read-only eval mode)
    → return Observation (no state change, small efficiency cost)

  {type: "transform", code: "df['Age'] = df['Age'].fillna(df['Age'].median())"}
    → save checkpoint (checkpoints/step_NNN.csv)
    → execute in persistent worker (AST-checked exec mode)
    → overwrite current.csv
    → grade(clean_df, result_df, error_map, metadata) → updated reward
    → return Observation

  {type: "undo"}
    → restore previous checkpoint from filesystem
    → send "reload" to worker to resync in-memory df
    → increment undo_count → re-grade with undo cost applied
    → return Observation

  {type: "validate"}
    → grade + build structured per-column / per-corruption-type diagnosis
    → increment validate_count (budget: 2 per episode)
    → return Observation(validate_result=structured_diagnosis)

  {type: "done"}
    → if first done AND score < 1.0:
        soft done → return score + remaining errors (done=False, agent continues)
      else:
        hard done → final grade, return Observation(done=True)

[Client]  client.py
  DataCleaningClient(base_url=ENV_URL)
    .reset(task_id=...) → StepResult
    .step(action)       → StepResult

[Inference]  inference.py
  for task in eval_task_ids:   # 25 pinned (dataset, difficulty, format) combos
    obs = env.reset(task_id)
    while not done:
      action = llm(system_prompt, obs)   # explore / transform / undo / validate / done
      obs = env.step(action)
```

## Component Responsibilities

| Component | Owns |
|-----------|------|
| `server/corruption/pipeline.py` | `CorruptionPipeline`: format selection, corruption orchestration, hints, diagnosis metadata |
| `server/corruption/value_corruptions.py` | 22 value-level corruption types (nulls, typos, type mangling, outliers, whitespace noise, swaps, …) |
| `server/corruption/format_corruptions.py` | 9 output formats and ~40 format-specific corruptions |
| `server/corruption/hints.py` | 3-level hint generation: strategy, tactical, categorical |
| `server/corruption/profiles.py` | Easy / medium / hard difficulty profiles |
| `server/corruption/categories.py` | 6 benchmark categories (FP / VR / MD / SR / SV / CP) |
| `server/rules/` | 7 semantic rule types, auto-inference from clean data, grading validation |
| `server/grader.py` | Multi-level reward formula, content-based row matching, collateral damage detection, action-cost penalties, semantic scoring |
| `server/environment.py` | Episode lifecycle: generative reset, action dispatch, checkpoint/undo coordination, observation building, `LEGACY_TASK_MAP` |
| `server/sandbox.py` | AST safety scanning, persistent worker management, raw-format file setup, CSV working copy, filesystem checkpoints, atexit cleanup |
| `server/worker.py` | Agent code execution with restricted builtins + pandas/numpy/io/openpyxl/yaml/lxml namespace, auto-rewrite `inplace=True` |
| `inference.py` | 25-task eval suite, LLM orchestration, prompt building, action parsing, auto-undo on regression |
| `tools/benchmark_runner.py` | CLI benchmark orchestrator: task matrix, inference runs, JSONL + CSV result persistence |
| `ui/` | Gradio dashboard: leaderboard, episode explorer, dataset catalog browser |
| `datasets/catalog.json` | 25 dataset entries with semantic rules, used by the generative environment |

## Grading

The grader receives `clean_df`, `result_df`, `error_map`, action counts, and optionally semantic rules. It never knows about corruption implementations — it scores result quality from first principles.

### Reward formula

```
# With semantic rules (auto-inferred per dataset):
base_score =
  schema_score       × 0.15   # column names, count, structural compatibility
  row_score          × 0.15   # drop/duplicate/reorder recovery via content-based matching
  cell_score         × 0.50   # per-cell accuracy (see below)
  distribution_score × 0.10   # coarse statistical similarity (mean, std, quantiles)
  semantic_score     × 0.10   # rule violation rate (7 constraint types)

# Without semantic rules:
base_score =
  schema_score       × 0.15
  row_score          × 0.20
  cell_score         × 0.55
  distribution_score × 0.10

efficiency_factor = max(0.5, 1.0 - transform_penalty - explore_penalty - undo_penalty - validate_penalty)
reward = base_score × efficiency_factor   [clamped to 0.0–1.0]
```

### Cell score mechanics

Cell score is the dominant component (50–55% of total). For each corrupted cell:

| Outcome | Score |
|---------|-------|
| Fixed (result == clean value) | 0 severity remaining |
| Accepted null fill (within 0.5σ of column range, matching accepted strategy) | 0 severity remaining |
| Numeric near-miss (≤5% relative error) | graduated partial credit |
| Unfixed (still dirty) | full severity remaining |
| Wrong value (changed to something incorrect) | **1.5× severity** |
| Collateral damage (correct cell the agent accidentally broke) | **+0.5 severity per cell** |

### Row matching

Row score uses **content-based matching** (not index-based). The grader builds a content-hash map of rows in both DataFrames and matches by content, handling:
- Dropped rows: credited if row is absent in both
- Duplicate rows: credited if eliminated
- Reordered rows: correctly aligned before comparison (no phantom collateral damage)
- CSV round-trip type normalization: `6.0 → "6"` so float/int variants match

### Efficiency costs

| Action | Cost |
|--------|------|
| Successful explore | 0.01 efficiency |
| Timed-out explore | 0.03 efficiency |
| Undo | configured `undo_cost` |
| Validate | configured `validate_cost` |
| Transforms beyond minimum | proportional penalty |

The 0.5 floor means a thorough explorer who fully cleans the data still gets at least 50% of the reward.

### Semantic scoring

7 constraint types are auto-inferred from each clean dataset and stored in `catalog.json`:

| Rule type | Checks |
|-----------|--------|
| Range | Numeric values within expected min/max |
| Regex | String values matching expected pattern |
| Enum | Categorical values within known set |
| Dtype | Column dtype matches expected type |
| NotNull | Non-nullable columns have no missing values |
| Unique | Unique-constraint columns have no duplicates |
| CrossColumn | Inter-column relationships (e.g. A ≤ B) |

## Sandbox Isolation

Each episode gets an isolated working directory:

```
outputs/sandbox/{episode_id}/
├── input.<format>     (original dirty raw file, never modified)
├── input.csv          (normalized dirty CSV)
├── current.csv        (live working copy, updated after each transform)
├── checkpoints/       (step_000.csv = dirty state; step_NNN.csv after each transform)
├── scripts/           (generated transform scripts for replay)
└── artifacts/         (execution outputs and snapshots)
```

Agent code runs in a persistent subprocess per episode with:

| Mechanism | Detail |
|-----------|--------|
| **AST safety scan** | Two modes: `exec` for transforms, `eval` for explores. Blocks `BLOCKED_MODULES` (`os`, `sys`, `subprocess`, `socket`, …), `BLOCKED_NAMES` (`open`, `exec`, `eval`, `getattr`, `globals`, `locals`, …), `BLOCKED_DUNDERS` (`__class__`, `__subclasses__`, `__getattribute__`, …) |
| **Restricted builtins** | Worker exec/eval namespace has an explicit allowlist (~30 safe functions); `open`, `exec`, `compile`, `object`, `issubclass`, `getattr`, `globals` absent |
| **Stripped environment** | Worker inherits only `PATH`, `HOME`, `LANG`, `LC_ALL`, `PYTHONPATH`, `VIRTUAL_ENV`; server secrets never forwarded |
| **Process group isolation** | `start_new_session=True`; timeout kills the full group |
| **Transform timeout** | 30 seconds |
| **Explore timeout** | 10 seconds |
| **Memory limit** | 2GB via `RLIMIT_AS` on Linux |
| **Stdout cap** | 1MB via `_BoundedStringIO` to prevent print-bomb OOM |
| **Persistent worker** | pandas/numpy loaded once per episode; transforms use `df.copy()` in exec namespace; explores use in-memory `df` directly (read-only) |
| **Undo sync** | Checkpoint restore sends `"reload"` to worker to resync in-memory df from `current.csv` |
| **Auto-rewrite inplace** | Worker converts `df['col'].fillna(val, inplace=True)` → `df['col'] = df['col'].fillna(val)` (pandas 2.x broke chained inplace) |
| **atexit cleanup** | `_active_workers` set + `atexit.register(_cleanup_all_workers)` in `sandbox.py`; `tini` in Dockerfile reaps zombies |

## File Formats

Agents may receive dirty data in any of 9 formats. The observation includes `file_format`, a `file_preview`, and `diagnosis` metadata. After the agent transforms `current.csv`, the grader compares it against the clean ground truth regardless of the original format.

| Format | Notes |
|--------|-------|
| CSV | Default; comma-delimited |
| TSV | Tab-delimited |
| JSON | Records or split orientation |
| JSONL | Newline-delimited records |
| Excel (.xlsx) | Read via openpyxl |
| XML | May have nesting/namespace errors |
| Fixed-width | Width mismatches as corruption |
| HTML table | Embedded in minimal HTML |
| SQL dump | INSERT statements; quote/escape issues |

## Difficulty Profiles

| Difficulty | Corruption types | Typical errors (Titanic) | Formats |
|------------|-----------------|--------------------------|---------|
| Easy | 1 focused type | 33–98 | CSV only |
| Medium | 3–4 types | 136–447 | 1–3 formats |
| Hard | 7–10 types | 1,586–2,895 | 3–5 formats |

Hard difficulty enables row-level operations (drop rows, duplicate rows, header injection into data), heavy format noise, and wide column spread. Medium caps column spread and uses lighter format noise.

## Eval Task Suite

`inference.py` runs 25 pinned (dataset, difficulty, format) combinations. Legacy task IDs (`titanic_easy`, etc.) are mapped through `LEGACY_TASK_MAP` in `environment.py` for backward compatibility.

| Dataset | Easy | Medium | Hard |
|---------|------|--------|------|
| Titanic | csv, tsv | csv, json | csv, json, xml |
| Iris | csv | csv, jsonl | — |
| Boston Housing | — | csv, json | csv, xml |
| Diabetes | — | csv, jsonl | csv, json |
| Wine Quality | csv | csv, json | csv, xml |
| Breast Cancer | csv | csv, jsonl | — |
