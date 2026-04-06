# Technical Documentation

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      inference.py                            │
│  LLM (OpenAI Client) ─── Agent Loop ─── [START/STEP/END]   │
└────────────────────────────┬────────────────────────────────┘
                             │ WebSocket
┌────────────────────────────▼────────────────────────────────┐
│                    FastAPI Server (app.py)                    │
│  create_app(DataCleaningEnvironment, ActionWrapper, ...)     │
├──────────────────────────────────────────────────────────────┤
│                  environment.py                               │
│  reset(task_id) → load task + dirty data → initial obs       │
│  step(action)   → explore | transform | done                 │
│  state          → episode metadata                           │
├──────────┬───────────────────────────┬───────────────────────┤
│ sandbox.py                           │ grader.py              │
│ AST safety scan                      │ Constraint checkers    │
│ subprocess + timeout                 │ Severity-weighted      │
│ Isolated working dir                 │ scoring                │
└──────────┴───────────────────────────┴───────────────────────┘
```

## Models

### Actions (discriminated union on `type` field)

**ExploreAction** (`type: "explore"`)
- `query: str` — pandas expression evaluated against `df`
- Read-only, no reward change
- Budget: 10 per transform cycle (resets after each transform)

**TransformAction** (`type: "transform"`)
- `code: str` — Python/pandas code operating on `df`
- Executed in subprocess sandbox
- Triggers re-grading and reward update

**DoneAction** (`type: "done"`)
- Ends the episode, triggers final grading

### Observation (DataCleaningObservation)

| Field | Type | Description |
|-------|------|-------------|
| `task_id` | str | Current task identifier |
| `task_description` | str | Human-readable task description |
| `constraints` | list[str] | Constraint descriptions |
| `data_summary` | str | Shape, dtypes, null counts, sample rows |
| `explore_result` | str? | Result of last explore |
| `transform_result` | str? | Success/error of last transform |
| `constraint_status` | dict[str, bool] | Constraint ID → satisfied |
| `reward` | float | Current severity-weighted score |
| `done` | bool | Episode terminated |
| `step_info` | StepInfo | Budget and step tracking |

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

Code submitted via `TransformAction` is executed in an isolated subprocess:

1. **AST safety scan** — blocks dangerous imports (os, sys, subprocess, socket, etc.) and calls (exec, eval, compile, open, __import__)
2. **Script generation** — preamble loads `df` from `current.csv`, postamble saves it back
3. **Subprocess execution** — `sys.executable` with 30s timeout
4. **Isolated directory** — `outputs/sandbox/{episode_id}/` with input.csv, current.csv, artifacts/, scripts/

**Allowed imports:** pandas, numpy, re, datetime, string, math, collections, itertools, functools, json, csv, os.path

## Grading

### Constraint Types

| Type | Check | Example |
|------|-------|---------|
| `no_nulls` | Column has zero nulls | `df['Age'].isnull().sum() == 0` |
| `dtype` | Column matches expected type | `float64`, `int64` |
| `no_duplicates` | No duplicate rows (optionally by key) | Full row or subset |
| `value_range` | All values in [min, max] | pH ∈ [2.0, 5.0] |
| `regex_match` | All values match pattern | Date format |
| `unique_values` | Values from allowed set | Embarked ∈ {S, C, Q} |
| `row_count_range` | Row count in [min, max] | ~1599 ± 10 |
| `no_whitespace` | No leading/trailing spaces | Names, tickets |
| `consistent_case` | Single casing pattern | All upper/lower/title |
| `cross_column` | Expression holds across columns | `end > start` |

### Scoring Formula

```
constraint_score = Σ(severity_i × solved_i) / Σ(severity_i)
efficiency_factor = max(0.5, 1.0 - (steps - min_steps) / (max_steps × 2))
reward = constraint_score × efficiency_factor
```

- `severity`: critical=3, high=2, medium=1
- Efficiency floors at 0.5 — quality always dominates
- Reward range: [0.0, 1.0]

## Corruption Engine (tools/corruption/engine.py)

Standalone tool, NOT deployed. Generates dirty data from clean sources.

| Corruption | Effect |
|-----------|--------|
| `inject_nulls` | Randomly null out cells |
| `type_mangle` | Mix string garbage ("N/A", "##") into numeric columns |
| `duplicate_rows` | Insert exact duplicates |
| `format_inconsistency` | Mix case styles (UPPER, lower, Title) |
| `whitespace_noise` | Leading/trailing/double spaces |
| `outlier_injection` | Extreme values (5-20σ from mean) |

## Task Config Schema

```json
{
  "task_id": "easy_titanic",
  "description": "...",
  "base_dataset": "titanic",
  "dirty_data_path": "data/dirty/easy_titanic.csv",
  "clean_data_path": "data/clean/titanic.csv",
  "constraints": [
    {
      "id": "c1",
      "type": "no_nulls",
      "column": "Age",
      "severity": "high",
      "description": "No null values in Age"
    }
  ],
  "min_transform_steps": 2,
  "max_transform_steps": 8
}
```
