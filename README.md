---
title: Data Cleaning OpenEnv
emoji: 🧹
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
default_tab: custom
---

# Data Cleaning OpenEnv Environment

An [OpenEnv](https://github.com/meta-pytorch/OpenEnv) environment where AI agents learn to clean dirty tabular data by writing Python/pandas code. Each episode is **generated dynamically** — the environment corrupts a clean dataset at reset time using a configurable corruption pipeline, then grades the agent's cleaning code against the ground truth using multi-level data quality scoring.

Built for the [Meta PyTorch OpenEnv Hackathon x Scaler School of Technology](https://www.scaler.com/school-of-technology/meta-pytorch-hackathon/).

## Quick Start

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -r server/requirements.txt

# Start the environment server locally
uvicorn server.app:app --port 8000 --ws-ping-interval 60 --ws-ping-timeout 120

# Run the baseline agent against all 25 eval tasks
python inference.py

# Run specific tasks (dataset name + difficulty + format)
python inference.py titanic_easy wine_medium
# or legacy style IDs
python inference.py titanic_easy titanic_medium titanic_hard

# Or run with Docker
docker build -t outlier-clean-data .
docker run -p 7860:7860 outlier-clean-data
```

## Architecture

The environment is fully generative — there are no pre-built dirty files. Each call to `/reset` runs the corruption pipeline fresh against a clean dataset.

```
[Datasets]
  datasets/catalog.json + materialized dataset files (6 datasets, 25 catalog entries)
       ↓
[Server: POST /reset]
  DataCleaningEnvironment.reset()
    → resolve task → load clean dataset + semantic rules from catalog
    → CorruptionPipeline(seed, difficulty).select_format()    [must precede corrupt()]
    → CorruptionPipeline.corrupt(clean_df, rules)
        ├─ value corruptions  (nulls, typos, type mangling, outliers, swaps, …)
        ├─ row/schema corruptions  (drop rows, duplicate rows, header injection)
        └─ format-specific corruptions  (delimiter noise, encoding shift, nesting errors, …)
    → create sandbox: write raw dirty file + normalized current.csv + step_000.csv checkpoint
    → grade(clean_df, current_df, error_map, metadata, rules) → initial reward
    → return Observation(file_format, file_preview, diagnosis, semantic_rules, hints)

[Server: POST /step]

  {type: "explore", query: "df.isnull().sum()"}
    → persistent worker (read-only), small efficiency cost per call
    → return Observation (no state change)

  {type: "transform", code: "df['Age'].fillna(df['Age'].median())"}
    → save checkpoint → execute in persistent worker (AST-checked)
    → overwrite current.csv → grade → updated reward
    → return Observation

  {type: "undo"}
    → restore previous checkpoint → increment undo_count → re-grade
    → return Observation

  {type: "validate"}
    → grade + build structured diagnosis (budgeted: 2 uses per episode)
    → return Observation(validate_result=structured_diagnosis)

  {type: "done"}
    → if first done AND score < 1.0:
        soft done → return score + remaining errors (done=False, agent continues)
      else:
        hard done → return final grade (done=True)

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
| `server/corruption/pipeline.py` | `CorruptionPipeline`: format selection, corruption orchestration, hints, diagnosis |
| `server/corruption/value_corruptions.py` | 22 value-level corruption types (nulls, typos, type mangling, outliers, whitespace noise, …) |
| `server/corruption/format_corruptions.py` | 9 output formats and ~40 format-specific corruptions |
| `server/corruption/profiles.py` | Easy / medium / hard difficulty profiles |
| `server/rules/` | 7 semantic rule types, auto-inference from clean data, grading validation |
| `server/grader.py` | Multi-level reward formula, content-based row matching, collateral damage, action-cost penalties |
| `server/environment.py` | Episode lifecycle: generative reset, action dispatch, checkpoint/undo coordination, observation building |
| `server/sandbox.py` | AST safety scanning, persistent worker management, raw-format setup, filesystem checkpoints |
| `server/worker.py` | Agent code execution with restricted builtins + pandas/numpy namespace, auto-rewrite `inplace=True` |
| `inference.py` | 25-task eval suite, LLM orchestration, prompt building, action parsing |
| `tools/benchmark_runner.py` | CLI orchestrator: task matrix, inference runs, JSONL + CSV result persistence |
| `ui/` | Gradio dashboard: leaderboard, episode explorer, dataset catalog |

## How It Works

1. **Reset**: Agent receives a dirty dataset in one of 9 file formats, plus: initial reward, error summary, file preview, corruption hints, and auto-inferred semantic rules for the dataset.
2. **Explore**: Agent inspects the data with read-only pandas queries (10 per episode, small cost per call).
3. **Transform**: Agent submits Python/pandas code executed in a sandboxed subprocess. The result is graded immediately.
4. **Undo / Validate**: Agent can revert to the last checkpoint (`undo`) or request structured error diagnostics (`validate`, 2 uses max).
5. **Done**: Agent signals completion. If score < 1.0 on the first done, it gets one retry (soft done) — it sees remaining errors and can continue. Perfect score or second done ends the episode.

## Action Space

| Action | Description |
|--------|-------------|
| `ExploreAction(query="df.isnull().sum()")` | Read-only pandas inspection, small efficiency cost |
| `TransformAction(code="df['Age'] = df['Age'].fillna(df['Age'].median())")` | Execute pandas code in sandbox |
| `UndoAction()` | Restore previous checkpoint, incurs undo cost |
| `ValidateAction()` | Get structured diagnosis without ending episode (2 uses max) |
| `DoneAction()` | Signal completion; first done is soft if score < 1.0 |

## Grading

The grader compares the agent's `current.csv` against the clean ground truth using a pre-generated error map. Rewards combine five quality dimensions:

```
# With semantic rules (auto-inferred per dataset):
base_score =
  schema_score       × 0.15   # column names, count, structural compatibility
  row_score          × 0.15   # duplicate/missing/reordered row recovery (content-based matching)
  cell_score         × 0.50   # per-cell accuracy: exact fixes, null fills, near-misses, penalties
  distribution_score × 0.10   # coarse statistical similarity (mean, std, quantiles)
  semantic_score     × 0.10   # rule violation rate (Range, Regex, Enum, Dtype, NotNull, Unique, CrossColumn)

# Without semantic rules (some tasks):
base_score =
  schema_score       × 0.15
  row_score          × 0.20
  cell_score         × 0.55
  distribution_score × 0.10

efficiency_factor = max(0.5, 1.0 - transform_penalty - explore_penalty - undo_penalty - validate_penalty)
reward = base_score × efficiency_factor   [clamped to 0.0–1.0]
```

### Cell scoring details

Cell score is the dominant component (50–55%). Key mechanics:

- **Fixed** (result == clean value) → full credit, 0 severity remaining
- **Accepted null fill** → full credit if fill is within 0.5σ of column range and matches the accepted strategy (`any` / `mean` / `median` / `mode` / `exact`)
- **Numeric near-miss** (≤5% relative error) → graduated partial credit
- **Wrong value** (changed to an incorrect value) → **1.5× severity penalty**
- **Collateral damage** (a correct cell the agent accidentally broke) → 0.5 severity penalty per cell
- **Unfixed** (still dirty) → full severity remaining

**Row matching is content-based**, not index-based. This correctly handles agents that drop duplicates, reorder rows, or reset indices — the grader matches rows by content hash rather than position.

### Efficiency costs

| Action | Cost |
|--------|------|
| Each successful explore | 0.01 efficiency |
| Each timed-out explore | 0.03 efficiency |
| Each undo | configured `undo_cost` |
| Each validate | configured `validate_cost` |
| Transforms beyond the minimum | proportional penalty |

The 0.5 floor on `efficiency_factor` means a thorough explorer who cleans everything still gets at least half the reward.

## File Formats

Agents may receive dirty data in any of 9 formats. The observation always includes `file_format`, a `file_preview`, and `diagnosis` metadata explaining what format-specific issues to expect.

| Format | Notes |
|--------|-------|
| CSV | Default; comma-delimited with optional quoting issues |
| TSV | Tab-delimited variant |
| JSON | Records or split orientation |
| JSONL | Newline-delimited JSON records |
| Excel (.xlsx) | Binary format; read via openpyxl |
| XML | Hierarchical; may have nesting/namespace errors |
| Fixed-width | Whitespace-aligned columns; width mismatches |
| HTML table | Embedded in minimal HTML; may have extra rows |
| SQL dump | INSERT statements; may have quote/escape issues |

After the agent transforms `current.csv`, the grader compares the result against the clean ground truth regardless of the original format. The normalize-to-CSV step is handled transparently by the sandbox.

## Difficulty Profiles

| Difficulty | Corruption types | Error count (Titanic) | Formats | Notes |
|------------|------------------|-----------------------|---------|-------|
| Easy | 1 focused type | 33–98 | CSV only | Single corruption family, light fraction |
| Medium | 3–4 types | 136–447 | 1–3 formats | Capped column spread, light format noise |
| Hard | 7–10 types | 1,586–2,895 | 3–5 formats | Row-level ops, heavy format noise, wide column spread |

## Eval Task Suite

`inference.py` runs a fixed 25-task suite of pinned (dataset, difficulty, format) combinations:

| Dataset | Easy | Medium | Hard |
|---------|------|--------|------|
| Titanic | csv, tsv | csv, json | csv, json, xml |
| Iris | csv | csv, jsonl | — |
| Boston Housing | — | csv, json | csv, xml |
| Diabetes | — | csv, jsonl | csv, json |
| Wine Quality | csv | csv, json | csv, xml |
| Breast Cancer | csv | csv, jsonl | — |

## Key Gotchas for LLM Agents

These are the most common failure modes observed when running LLM agents:

**`pd.to_numeric(errors='coerce')` introduces NaN which the grader penalizes 1.5×.** If the clean value is a real number (e.g. `0`) and the agent converts the sentinel `"##"` to `NaN` instead of the correct value, the cell is scored as a *wrong value*, not a fix. Always impute a real value in the same transform:
```python
# Bad: NaN where clean value is 0
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')

# Good: coerce then fill in one step
df['Age'] = pd.to_numeric(df['Age'], errors='coerce').fillna(df['Age'].median())
```

**Sentinel strings like `"n/a"`, `"NA"`, `"null"` are auto-parsed to `NaN` by `pd.read_csv()`.** The agent sees `NaN` in the DataFrame even though the dirty file had a string sentinel. The observation's `diagnosis` metadata explains this.

**Never drop rows you didn't corrupt.** Collateral damage is tracked — correctly-valued rows that disappear add penalty.

**Wrong-value penalty is 1.5×.** It's better to leave a value dirty (full severity) than to change it to the wrong value (1.5× severity).

## Sandbox Isolation

Each episode gets an isolated working directory:

```
outputs/sandbox/{episode_id}/
├── input.<format>     (original dirty raw file, never modified)
├── input.csv          (normalized dirty CSV working copy)
├── current.csv        (live working copy, updated after each transform)
├── checkpoints/       (step_000.csv = dirty state; step_NNN.csv after each transform)
├── scripts/           (generated transform scripts for replay)
└── artifacts/         (execution outputs and snapshots)
```

Agent code runs in a subprocess with:

- **AST safety scan** — blocks `os`, `sys`, `subprocess`, `socket`, and 8 other modules; blocks `open`, `exec`, `eval`, `getattr`, `globals`, `locals`, `compile`, and dangerous dunders (`__class__`, `__subclasses__`, `__getattribute__`, …)
- **Restricted builtins** — worker exec/eval namespace uses an explicit allowlist (~30 safe functions); `open`, `exec`, `compile`, `object`, `issubclass`, and `getattr` are absent
- **Stripped environment** — worker subprocess only inherits `PATH`, `HOME`, `LANG`, `LC_ALL`, `PYTHONPATH`, `VIRTUAL_ENV`; server secrets (API keys, HF_TOKEN) are never forwarded
- **Process isolation** — `start_new_session=True`; timeout kills the full process group
- **30s transform timeout, 10s explore timeout**
- **2GB memory limit** (`RLIMIT_AS` on Linux)
- **1MB stdout cap** (`_BoundedStringIO`) to prevent print-bomb OOM
- **Persistent worker** — pandas/numpy loaded once per episode; undo sends a `"reload"` command to resync the worker's in-memory DataFrame after checkpoint restore
- **Auto-rewrite `inplace=True`** — worker converts `df['col'].fillna(val, inplace=True)` → `df['col'] = df['col'].fillna(val)` automatically (pandas 2.x broke chained inplace)

## Benchmark Status

The old benchmark tables were measured before the April 2026 difficulty rebalance and are no longer comparable. The generator now uses a more gradual curve with capped column spread at medium difficulty. Full model benchmarks should be rerun.

### Post-rebalance profile sanity check (Titanic, CSV, seeds 42–51)

| Difficulty | Typical errors | Average errors | Notes |
|------------|----------------|----------------|-------|
| Easy | 33–98 | 61.5 | Single focused corruption type |
| Medium | 136–447 | 278.0 | 3–4 types, capped column spread |
| Hard | 1,586–2,895 | 2,174.0 | Successful seeds only; some seeds still hit a dtype bug |

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `API_BASE_URL` | `https://api.openai.com/v1` | LLM endpoint (validator injects LiteLLM proxy here) |
| `API_KEY` | from `HF_TOKEN` or `OPENAI_API_KEY` | API token, read at import time |
| `MODEL_NAME` | `gpt-4o` | Model name |
| `ENV_URL` | `http://localhost:7860` | OpenEnv server URL |
| `LOG_LEVEL` | `INFO` | `INFO` = actions/timing; `DEBUG` = full LLM I/O |
| `LOG_DIR` | `outputs/logs` | JSONL log directory |
| `MIN_CALL_INTERVAL` | `2.5` | Min seconds between LLM calls (set to 0 for local) |
| `SANDBOX_BASE` | `outputs/sandbox` | Episode working directories |
| `ENABLE_WEB_INTERFACE` | unset | Set to enable Gradio dashboard at `/web` |

## Project Structure

```
models.py                    # Pydantic types: actions, observations, state, error map
client.py                    # WebSocket EnvClient subclass
inference.py                 # Baseline LLM agent — 25-task eval suite
server/
  app.py                     # FastAPI application
  environment.py             # Core reset/step/state loop, generative episodes, LEGACY_TASK_MAP
  sandbox.py                 # AST safety, persistent worker management, filesystem checkpoints
  worker.py                  # Worker process: exec/eval with restricted builtins, inplace rewriting
  grader.py                  # Multi-level reward formula, content-based row matching, semantic scoring
  corruption/
    pipeline.py              # CorruptionPipeline: format selection + corruption orchestration
    value_corruptions.py     # 22 value-level corruption types
    format_corruptions.py    # 9 file formats + ~40 format-specific corruptions
    hints.py                 # 3-level hint generation (strategy / tactical / categorical)
    profiles.py              # Easy / medium / hard difficulty profiles
    categories.py            # 6 benchmark categories (FP / VR / MD / SR / SV / CP)
  rules/
    types.py                 # 7 semantic rule types and validation logic
    infer.py                 # Auto-infer rules from clean DataFrames
    enrich_catalog.py        # Batch-enrich catalog.json with inferred rules
datasets/
  catalog.json               # 25 dataset entries used by the generative environment
tools/
  download_datasets.py       # Dataset download pipeline
  benchmark_runner.py        # CLI benchmark orchestrator
  benchmark_config.yaml      # Default benchmark config
ui/
  app.py                     # Gradio dashboard entry point
  leaderboard.py             # Model × category leaderboard
  explorer.py                # Step-by-step episode replay viewer
  catalog_view.py            # Dataset catalog browser with rule viewer
  data_loader.py             # Loads benchmark results and episode logs
docs/                        # Architecture, ADRs, deployment runbook
```

## Documentation

- [docs/architecture.md](docs/architecture.md) — Full data flow, component responsibilities, grading formula, sandbox details
- [docs/technical.md](docs/technical.md) — Deep dive on multi-level scoring, semantic rules, corruption system
- [docs/deployment.md](docs/deployment.md) — Local dev, Docker, HF Spaces, pre-submission checklist
