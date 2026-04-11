# Data Cleaning OpenEnv Environment — Project Context

## Purpose

OpenEnv environment for the Meta PyTorch Hackathon (deadline: April 8, 2026). AI agents clean dynamically corrupted tabular data by writing Python data-cleaning code, graded against the ground truth with multi-level data quality scoring and action costs.

## Architecture

- **server/environment.py** — Core Environment class; generates corrupted episodes at `reset()` via `CorruptionPipeline`; supports `explore`, `transform`, `done`, checkpoint-backed `undo`, and budgeted `validate`
- **server/sandbox.py** — Persistent worker process per episode; accepts dirty content and file format, writes raw input plus CSV working copy, and exposes filesystem-backed checkpoints for undo
- **server/worker.py** — Worker process: re-reads CSV each step, auto-rewrites `inplace=True` patterns, exposes pandas/numpy plus `io`, `openpyxl`, `yaml`, and `lxml`
- **server/grader.py** — Multi-level grader: schema, row, cell, distribution, and semantic scores with content-based row matching and action-cost parameters
- **server/app.py** — FastAPI via `openenv.core.create_app()`, health endpoint at `/`
- **Dockerfile** (root) — HF Spaces deployment on port 7860; builds from `pyproject.toml`, installs lxml system deps, copies datasets/tools
- **models.py** — Pydantic types: ExploreAction, TransformAction, DoneAction, UndoAction, ValidateAction, ErrorMap, CellError, RowError, Observation, State
- **client.py** — EnvClient subclass (WebSocket)
- **inference.py** — Baseline agent using any OpenAI-compatible API; 15-task eval suite, undo/validate support, enriched prompt fields (`file_format`, `file_preview`, `diagnosis`, `validate_result`)
- **server/corruption/** — Runtime corruption subsystem: `CorruptionPipeline.select_format()` + `corrupt()`, 22 value corruptions, multi-format raw inputs, ~40 format-specific corruptions, 3 difficulty profiles, 3-level hints, 6 benchmark categories
- **server/corruption/categories.py** — 6 benchmark categories (FP/VR/MD/SR/SV/CP) mapping to corruption subsets and format pools
- **server/rules/** — 7 semantic rule types (Range, Regex, Enum, Dtype, NotNull, Unique, CrossColumn), auto-inferred from clean data, validated in grading
- **datasets/** — 25-entry dataset catalog plus download pipeline in `tools/download_datasets.py`
- **tools/benchmark_runner.py** — CLI benchmark orchestrator: generates task matrix, runs inference agent, saves JSONL + CSV results
- **ui/** — Gradio benchmark dashboard: leaderboard, episode explorer, dataset catalog tabs

## Key Decisions

- **Code generation over structured transforms** — agents write real pandas code
- **Generative episodes over static tasks** — `reset()` samples a dataset/profile/format and creates corruptions dynamically; `LEGACY_TASK_MAP` preserves old task IDs for compatibility
- **Multi-format input** — agents may receive csv, json, jsonl, excel, tsv, xml, fixed-width, html table, sql dump, or yaml with file previews and diagnosis metadata in observations
- **Multi-level grading over simple diff scoring** — reward combines schema_score (0.15), row_score (0.15/0.20), cell_score (0.50/0.55), distribution_score (0.10), and semantic_score (0.10 when rules present)
- **Content-based row matching** — row recovery is matched by content rather than only by index, improving resilience to reordering and format round-trips
- **Wrong-value penalty** — changing a cell to an incorrect value is penalized 1.5×; numeric near-misses (≤5% relative error) get graduated partial credit instead of full penalty
- **Collateral damage penalty** — cells that were correct but got corrupted by the agent add 0.5 severity each
- **Explore cost** — each explore action incurs a small efficiency penalty (0.01/step, 0.03 for timeouts), discouraging excessive or wasteful exploration
- **Undo cost** — undo restores the last filesystem checkpoint and applies a configured score cost through `undo_count`
- **Validate budget** — agents get 2 validate actions per episode; validation returns structured diagnosis without ending the episode and tracks `validate_count`/`validate_uses`
- **Soft done** — first done is a checkpoint if reward < 1.0: agent sees score + remaining errors and can continue. Second done is final. Perfect score always ends immediately. Capped at 1 retry
- **Null fill tolerance** — `accepted_fill` field per null error: "any" accepts any reasonable imputation (mean, median, mode, ffill, bfill) within 0.5σ of column range, "exact" requires the clean value, "mean"/"median"/"mode" accept only that specific statistic
- **Persistent worker** — pandas/numpy loaded once per episode; transform uses `df.copy()` in exec namespace (no CSV re-read per step); explore uses in-memory `df` directly (read-only); undo sends a `"reload"` command to resync worker state after checkpoint restore
- **Auto-rewrite inplace=True** — worker auto-converts `df['col'].fillna(val, inplace=True)` → `df['col'] = df['col'].fillna(val)` (pandas 2.x broke chained inplace)
- **Difficulty profiles** — easy/medium/hard profiles now increase more gradually: easy stays focused, medium uses 3-4 corruption families with capped column spread and light format noise, and hard keeps the wide row/schema/format mix
- **Format-aware corruption safety** — `CorruptionPipeline.select_format()` is called before `corrupt()` so raw dirty content, parser expectations, hints, and previews stay aligned
- **Inference config** — `inference.py` reads `API_BASE_URL`, `MODEL_NAME`, and `API_KEY` (from `HF_TOKEN`/`OPENAI_API_KEY`) at import time via `os.environ.get`; validator injects LiteLLM proxy values which take precedence

## Conventions

- OpenEnv spec v1: typed models, `reset()`/`step()`/`state()` API
- Dual-import in server files: `try: from ..models / except: from models`
- Rewards in 0.0–1.0 range, with schema/row/cell/distribution components plus action costs
- `inference.py` LLM calls use `OpenAI(base_url=API_BASE_URL, api_key=API_KEY)` — env vars read once at import
- Client never imports server
- Sandbox always on for code execution
- Corruption subsystem owns task generation, format selection, hints, and raw dirty content

## Invariants

- Rewards always in [0.0, 1.0]
- Sandbox always on — no code execution without AST scan + persistent worker
- Grader never knows about corruption implementations — it scores result quality from clean data, error maps, and generated metadata
- Grader detects collateral damage (correct cells the agent broke) using content-based row mapping so reordered rows are aligned before comparison
- Dynamic reset produces clean data, dirty raw content, normalized CSV working copy, error map, metadata, hints, and observation previews
- CSV is the source of truth between steps; worker in-memory `df` is kept in sync (reload on undo)
- Client never imports server
- Dual-import pattern in server files: `try: from ..models / except: from models`

## Commands

```bash
source .venv/bin/activate
# Run server (with WebSocket keepalive for long LLM calls)
uvicorn server.app:app --port 8000 --ws-ping-interval 60 --ws-ping-timeout 120
# Run agent
python inference.py
# Run specific tasks
python inference.py titanic_easy wine_hard
# Download/update dataset catalog inputs
python tools/download_datasets.py
```

## Benchmark Status

The old model benchmark tables were captured before the April 2026 difficulty rebalance and are no longer comparable to the current generator. Do not use the old `titanic_medium`/`wine_medium` reward numbers as current baselines.

### Post-rebalance profile sanity check (Titanic, CSV, seeds 42-51)

| Difficulty | Typical total errors | Average total errors | Notes                                                                  |
| ---------- | -------------------- | -------------------- | ---------------------------------------------------------------------- |
| easy       | 33-98                | 61.5                 | Single focused corruption type                                         |
| medium     | 136-447              | 278.0                | 3-4 corruption types with capped column spread                         |
| hard       | 1586-2895            | 2174.0               | Successful seeds only; hard-mode dtype bug still present on some seeds |

Full model benchmarks should be rerun after the rebalance before publishing new reward baselines.

## Deployment

- **HF Space**: https://huggingface.co/spaces/shivshri/openenv_dataclean
- **Dockerfile** at repo root for HF Spaces (Docker SDK)
- Health endpoint: `GET /` returns 200
- Runtime: ~5.4 min for all 6 tasks on 2 vCPU / 8GB

## Known Issues

- Small models (2B) give up early on hard tasks — larger models (70B+ via Groq) should do better
- Context window overflow on hard tasks with small local models (8192 tokens) — inference auto-submits `done` on context exceeded
- `titanic_hard` shows 16/958 fixed at reset — phantom matches from overlapping corruptions where dirty value happens to equal clean value
- ~~`clean_value: None` overlap bug~~ — fixed: `_get_clean_val()` always reads from original clean_df, not corrupted df
- Grader index type mismatch: if `result_df` has string index (e.g. after CSV round-trip) but error_map keys are int-based, `df.at[int_idx, col]` silently KeyErrors → all cells marked "unfixed". Pre-existing bug, not yet fixed
- ~~Grader `_row_hash` int/float mismatch~~ — fixed: CSV round-trip converts int columns to float when NaN is present, causing `str(6) ≠ str(6.0)` and zero row matches in `match_rows_by_content`. `_row_hash` now normalizes numeric values (e.g. `6.0` → `"6"`). This was harmless for same-row-count tasks (identity fallback) but broke hard tasks with `drop_rows`/`duplicate_rows`
- Medium/hard 0-progress with small models: not an environment bug — `pd.to_numeric(errors='coerce')` converts sentinels like `"##"` → NaN when clean value is `0`, graded as `wrong_value` (1.5× penalty). Models must impute correct values, not just coerce to NaN. Type_mangle sentinels like "n/a"/"N/A" are auto-parsed as NaN by `pd.read_csv()` before the agent sees them (`dirty_value=None`)
- ~~Collateral damage detection could compare dropped rows against shifted result indices~~ — fixed: it is row-mapping aware and skips clean rows missing from the content mapping after row drops

## File Map

- `models.py` — Pydantic action, observation, state, error-map, undo, and validate types
- `server/environment.py` — Generative env loop, `LEGACY_TASK_MAP`, action dispatch, observation building
- `server/sandbox.py` — Persistent worker management, raw file/CSV setup, AST safety (exec + eval modes), filesystem checkpoints, env stripping, atexit cleanup
- `server/worker.py` — Worker process: exec/eval agent code with restricted `__builtins__`, inplace rewriting, `_BoundedStringIO` stdout cap, 2GB memory limit (Linux), reload-on-undo
- `server/grader.py` — Multi-level reward formula, semantic scoring, and validation diagnostics
- `server/app.py` — FastAPI wiring
- `client.py` — WebSocket client
- `inference.py` — LLM agent baseline (model-agnostic), 15-task eval suite
- `server/corruption/pipeline.py` — Runtime `CorruptionPipeline`, format selection, corruption orchestration
- `server/corruption/value_corruptions.py` — 22 value-level corruption types
- `server/corruption/format_corruptions.py` — 9 file formats and format-specific corruptions
- `server/corruption/hints.py` — Strategy, tactical, and categorical hint generation
- `server/corruption/profiles.py` — Easy/medium/hard corruption profiles
- `server/corruption/categories.py` — 6 benchmark categories (FP/VR/MD/SR/SV/CP)
- `server/rules/types.py` — 7 semantic rule types and validation logic
- `server/rules/infer.py` — Auto-infer rules from clean DataFrames
- `server/rules/enrich_catalog.py` — Batch-enrich catalog.json with inferred rules
- `datasets/catalog.json` — 25 dataset entries used by the generative environment
- `tools/download_datasets.py` — Dataset download pipeline (GitHub mirrors + source URLs)
- `tools/benchmark_runner.py` — CLI benchmark orchestrator for model evaluation
- `tools/benchmark_config.yaml` — Default benchmark config (models, categories, difficulties)
- `ui/app.py` — Gradio dashboard entry point
- `ui/leaderboard.py` — Model × Category leaderboard pivot table
- `ui/explorer.py` — Step-by-step episode replay viewer
- `ui/catalog_view.py` — Dataset catalog browser with rule viewer
- `ui/data_loader.py` — Loads benchmark results, episode logs, catalog

## Task IDs

- `inference.py` defines a 15-task eval suite in `EVAL_TASK_IDS`
- Legacy IDs such as `titanic_easy`, `titanic_medium`, `titanic_hard`, `wine_easy`, `wine_medium`, and `wine_hard` are mapped through `LEGACY_TASK_MAP` for backward compatibility

## Grading Formula

```
# With semantic rules (auto-inferred from clean data):
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

cell_score accounts for exact fixes, accepted null fills, numeric near-misses,
wrong-value penalties, whitespace/format tolerance, and collateral damage.
row_score uses content-based matching for dropped/duplicated/reordered rows.
semantic_score measures rule violation rate against 7 auto-inferred constraint types.

transform_penalty = max(0, transform_steps - min_steps) / (max_steps × 2)
explore_penalty   = (successful_explores × 0.01) + (timed_out_explores × 0.03)
undo_penalty      = undo_count × undo_cost
validate_penalty  = validate_count × validate_cost
efficiency_factor = max(0.5, 1.0 - transform_penalty - explore_penalty - undo_penalty - validate_penalty)
reward = base_score × efficiency_factor   [clamped to 0.0–1.0]
```

## Environment Variables

| Var                 | Default                             | Purpose                                             |
| ------------------- | ----------------------------------- | --------------------------------------------------- |
| `API_BASE_URL`      | `https://api.openai.com/v1`         | LLM endpoint; validator injects LiteLLM proxy here  |
| `API_KEY`           | from `HF_TOKEN` or `OPENAI_API_KEY` | API token (read at import time)                     |
| `MODEL_NAME`        | `gpt-4o`                            | Model name                                          |
| `ENV_URL`           | `http://localhost:7860`             | OpenEnv server URL                                  |
| `LOG_LEVEL`         | `INFO`                              | `INFO` for actions/timing, `DEBUG` for full LLM I/O |
| `LOG_DIR`           | `outputs/logs`                      | JSONL log directory                                 |
| `MIN_CALL_INTERVAL` | `2.5`                               | Min seconds between LLM calls (0 for local)         |
| `TASKS_DIR`         | `tasks`                             | Task config directory                               |
| `DATA_DIR`          | `data`                              | Data artifacts directory                            |
| `DATASETS_DIR`      | `datasets`                          | Dataset catalog/materialized inputs                 |
| `SANDBOX_BASE`      | `outputs/sandbox`                   | Sandbox working directories                         |

## Deployment Notes

- Validator/deployed runs should rely on the injected LiteLLM proxy values when they are present.
- The current script defaults `API_BASE_URL` to `https://api.openai.com/v1` and `MODEL_NAME` to `gpt-4o` when unset.
- `API_KEY` is read from `HF_TOKEN` then `OPENAI_API_KEY` at import time (empty string if neither set).
