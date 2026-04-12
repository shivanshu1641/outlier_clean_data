# Technical Documentation

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      inference.py                            │
│  LLM (OpenAI Client) ─── Agent Loop ─── [START/STEP/END]   │
│  Auto-transforms → LLM transforms → Auto-undo on regress   │
└────────────────────────────────┬────────────────────────────┘
                                 │ WebSocket
┌────────────────────────────────▼────────────────────────────┐
│                    FastAPI Server (app.py)                    │
│  create_app(DataCleaningEnvironment, ActionWrapper, ...)     │
├──────────────────────────────────────────────────────────────┤
│                  environment.py                               │
│  reset(task_id) → load task + dirty data → initial obs       │
│  step(action)   → explore | transform | undo | validate |    │
│                    done                                       │
│  state          → episode metadata                           │
├──────────────┬───────────────────────────┬───────────────────┤
│ sandbox.py   │ worker.py                 │ grader.py          │
│ AST safety   │ Persistent subprocess     │ Multi-level        │
│ Checkpoints  │ Restricted builtins       │ scoring + content  │
│ Isolated dir │ inplace rewrite           │ row matching       │
└──────────────┴───────────────────────────┴───────────────────┘
```

## Models

### Actions (discriminated union on `type` field)

**ExploreAction** (`type: "explore"`)
- `query: str` — pandas expression evaluated against `df`
- Read-only, no state change
- Budget: 10 per transform cycle (resets after each transform)
- Cost: 0.01 efficiency per successful explore, 0.03 for timeouts

**TransformAction** (`type: "transform"`)
- `code: str` — Python/pandas code operating on `df`
- Executed in persistent worker subprocess with AST safety scan
- Triggers re-grading and reward update
- Checkpoint saved before execution for undo

**UndoAction** (`type: "undo"`)
- Restores the previous filesystem checkpoint
- Sends `"reload"` to worker to resync in-memory df
- Incurs `undo_cost` efficiency penalty
- Increments `undo_count`

**ValidateAction** (`type: "validate"`)
- Returns structured per-column, per-corruption-type diagnosis
- Budget: 2 per episode
- Does not end the episode
- Incurs `validate_cost` efficiency penalty

**DoneAction** (`type: "done"`)
- First done (if reward < 1.0): soft done — returns score + remaining errors, agent continues
- Second done or perfect score: hard done — final grade, episode ends
- Capped at 1 retry

### Observation (DataCleaningObservation)

| Field | Type | Description |
|-------|------|-------------|
| `task_id` | str | Current task identifier |
| `task_description` | str | Human-readable task description |
| `file_format` | str | Input file format (csv, json, jsonl, etc.) |
| `file_preview` | str | Raw file content preview |
| `diagnosis` | dict | Corruption metadata from pipeline |
| `constraints` | list[str] | Semantic rule descriptions |
| `data_summary` | str | Shape, dtypes, null counts, sample rows |
| `explore_result` | str? | Result of last explore |
| `transform_result` | str? | Success/error of last transform |
| `validate_result` | dict? | Structured validation diagnosis (when validate used) |
| `constraint_status` | dict[str, bool] | Constraint ID → satisfied |
| `reward` | float | Current severity-weighted score |
| `done` | bool | Episode terminated |
| `step_info` | StepInfo | Budget and step tracking |
| `suggested_explore_queries` | list[str] | Hints for useful explore queries |
| `remaining_errors_by_type` | dict | Error counts by corruption type |

### State (DataCleaningState)

| Field | Type |
|-------|------|
| `episode_id` | str |
| `task_id` | str |
| `step_count` | int |
| `explore_steps_total` | int |
| `transform_steps_total` | int |
| `current_reward` | float |
| `constraints_satisfied` | int |
| `constraints_total` | int |

## Sandbox

Code submitted via `TransformAction` runs in an isolated persistent worker subprocess:

1. **AST safety scan** — two modes: `exec` for transforms, `eval` for explores. Blocks `BLOCKED_MODULES` (os, sys, subprocess, socket, …), `BLOCKED_NAMES` (open, exec, eval, getattr, globals, locals, …), `BLOCKED_DUNDERS` (__class__, __subclasses__, __getattribute__, …)
2. **Persistent worker** — pandas/numpy loaded once per episode; transforms use `df.copy()` in exec namespace; explores use in-memory `df` directly (read-only)
3. **Restricted builtins** — explicit allowlist (~30 safe functions); `open`, `exec`, `compile`, `object`, `issubclass`, `getattr`, `globals` absent
4. **Allowed imports:** pandas, numpy, re, datetime, string, math, collections, itertools, functools, json, csv, os.path, io, openpyxl, yaml, lxml
5. **Isolated directory** — `outputs/sandbox/{episode_id}/` with input.<format>, input.csv, current.csv, checkpoints/, scripts/, artifacts/
6. **Checkpoint/undo** — filesystem checkpoints at `checkpoints/step_NNN.csv`; undo restores previous checkpoint and sends `"reload"` to worker
7. **Auto-rewrite inplace** — converts `df['col'].fillna(val, inplace=True)` → `df['col'] = df['col'].fillna(val)` (pandas 2.x broke chained inplace)
8. **Timeouts** — transforms: 30s, explores: 10s
9. **Memory limit** — 2GB via `RLIMIT_AS` on Linux
10. **Stdout cap** — 1MB via `_BoundedStringIO` to prevent print-bomb OOM
11. **Process isolation** — `start_new_session=True`; timeout kills the full process group
12. **Stripped environment** — worker inherits only `PATH`, `HOME`, `LANG`, `LC_ALL`, `PYTHONPATH`, `VIRTUAL_ENV`
13. **Cleanup** — `_active_workers` set + `atexit.register(_cleanup_all_workers)`; `tini` in Dockerfile reaps zombies

## Grading

### Reward Formula

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

transform_penalty = max(0, transform_steps - min_steps) / (max_steps × 2)
explore_penalty   = (successful_explores × 0.01) + (timed_out_explores × 0.03)
undo_penalty      = undo_count × undo_cost
validate_penalty  = validate_count × validate_cost
efficiency_factor = max(0.5, 1.0 - transform_penalty - explore_penalty - undo_penalty - validate_penalty)
reward = base_score × efficiency_factor   [clamped to 0.0–1.0]
```

### Cell Score Mechanics

Cell score is the dominant component (50–55% of total). For each corrupted cell:

| Outcome | Score |
|---------|-------|
| Fixed (result == clean value) | 0 severity remaining |
| Accepted null fill (within 0.5σ of column range, matching accepted strategy) | 0 severity remaining |
| Numeric near-miss (≤5% relative error) | graduated partial credit |
| Unfixed (still dirty) | full severity remaining |
| Wrong value (changed to something incorrect) | **1.5× severity** |
| Collateral damage (correct cell the agent accidentally broke) | **+0.5 severity per cell** |

### Row Matching

Row score uses **content-based matching** (not index-based). The grader builds a content-hash map of rows in both DataFrames and matches by content, handling:
- Dropped rows: credited if row is absent in both
- Duplicate rows: credited if eliminated
- Reordered rows: correctly aligned before comparison (no phantom collateral damage)
- Cross-type numeric normalization: `6.0`, `"6.0"`, `6` all normalize to `"6"` so float/int/str variants match after CSV round-trips and format parsing

### Semantic Scoring

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

## Corruption Engine

The corruption pipeline (`server/corruption/pipeline.py`) generates dirty data dynamically at `reset()` time. There are no pre-built dirty files.

### Value Corruptions (22 types)
inject_nulls, type_mangle, duplicate_rows, format_inconsistency, whitespace_noise, outlier_injection, value_swap, sentinel_injection, precision_loss, unit_inconsistency, encoding_corruption, case_corruption, date_format_mixing, numeric_string_mixing, boolean_inconsistency, categorical_drift, measurement_error, rounding_error, truncation, padding, scientific_notation, business_rule_violation

### Row/Schema Corruptions
drop_rows, duplicate_rows, header_injection

### Format-Specific Corruptions (~40 types across 9 formats)
Delimiter noise, encoding shifts, nesting errors, quote/escape issues, width mismatches, etc.

### Difficulty Profiles

| Difficulty | Corruption types | Typical errors (Titanic) | Formats |
|------------|-----------------|--------------------------|---------|
| Easy | 1 focused type | 33–98 | CSV only |
| Medium | 3–4 types | 136–447 | 1–3 formats |
| Hard | 7–10 types | 1,586–2,895 | 3–5 formats |

### Benchmark Categories

| ID | Name | Description |
|----|------|-------------|
| FP | Format & Parsing | Malformed delimiters, encoding, whitespace |
| VR | Value Range | Out-of-range values, unit inconsistencies |
| MD | Missing Data | Nulls, empty strings, missing rows |
| SR | Schema & References | Wrong dtypes, referential violations |
| SV | Semantic Violations | Business rule violations, cross-column errors |
| CP | Comprehensive | All corruption types combined |

## Inference Agent

### Auto-Transforms

Before the LLM loop begins, the agent applies corruption-type-aware auto-transforms:

| Template | Action |
|----------|--------|
| `duplicate_rows` | `df.drop_duplicates(inplace=True)` |
| `inject_nulls` | `fillna(median)` for numeric, `fillna(mode)` for categorical |
| `whitespace_noise` | `.str.strip()` on object columns |

These provide a strong baseline (easy tasks often reach 0.80–0.95 from auto-transforms alone).

### Auto-Undo

When a transform causes ≥25% regression from the best reward, the agent automatically:
1. Reverts to the best checkpoint
2. Preserves `best_reward` (does NOT reset to post-undo value, which includes undo penalty)
3. Injects a warning into the next prompt with the failed column names, advising the model to target different columns/corruption types

### Step Limits

| Difficulty | Max steps |
|------------|-----------|
| Easy | 30 |
| Medium | 60 |
| Hard | 100 |
