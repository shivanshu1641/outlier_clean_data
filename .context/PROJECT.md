# Data Cleaning OpenEnv Environment — Project Context

## Purpose

OpenEnv environment for the Meta PyTorch Hackathon (deadline: April 8, 2026). AI agents clean dynamically corrupted tabular data by writing Python data-cleaning code, graded against the ground truth with multi-level data quality scoring and action costs.

## Architecture

- **server/environment.py** — Core Environment class; generates corrupted episodes at `reset()` via `CorruptionPipeline`; supports `explore`, `transform`, `done`, checkpoint-backed `undo`, and budgeted `validate`; contains `LEGACY_TASK_MAP` for backward-compatible task IDs
- **server/sandbox.py** — Persistent worker process per episode; accepts dirty content and file format; writes raw input file plus normalized CSV working copy; exposes filesystem-backed checkpoints for undo; atexit cleanup of all active workers
- **server/worker.py** — Worker process: re-reads CSV each step, auto-rewrites `inplace=True` patterns, exposes pandas/numpy plus `io`, `openpyxl`, `yaml`, and `lxml`; restricted `__builtins__` allowlist; `_BoundedStringIO` stdout cap; 2GB memory limit (Linux); `"reload"` command resyncs in-memory df after undo
- **server/grader.py** — Multi-level grader: schema, row, cell, distribution, and semantic scores; content-based row matching with cross-type numeric normalization (`6.0`/`"6.0"`/`6` all normalize to `"6"`); collateral damage detection via row_mapping; action-cost parameters; `accepted_fill` logic with 0.5σ margin
- **server/app.py** — FastAPI via `openenv.core.create_app()`, health endpoint at `/`, optional Gradio dashboard at `/web` via `ENABLE_WEB_INTERFACE`
- **Dockerfile** (root) — HF Spaces deployment on port 7860; builds from `pyproject.toml`, installs lxml system deps, copies datasets/tools; `tini` for zombie reaping
- **models.py** — Pydantic types: ExploreAction, TransformAction, DoneAction, UndoAction, ValidateAction, ErrorMap, CellError, RowError, Observation, State
- **client.py** — EnvClient subclass (WebSocket)
- **inference.py** — Baseline agent using any OpenAI-compatible API; 18-task eval suite (6 datasets × 3 per dataset, pinned difficulty+format triples); auto-transforms (duplicate_rows, inject_nulls with median/mode, whitespace_noise); undo/validate support; auto-undo on 25%+ regression with best_reward preservation and post-undo warnings; auto-validate on budget exhaustion; enriched prompt fields (`file_format`, `file_preview`, `diagnosis`, `validate_result`, `suggested_explore_queries`, `remaining_errors_by_type`)
- **server/corruption/pipeline.py** — Runtime `CorruptionPipeline`: `select_format()` must be called before `corrupt()`; 22 value corruptions; multi-format raw inputs; ~40 format-specific corruptions; 3 difficulty profiles; 3-level hints; 6 benchmark categories
- **server/corruption/categories.py** — 6 benchmark categories (FP/VR/MD/SR/SV/CP) mapping to corruption subsets and format pools
- **server/rules/** — 7 semantic rule types (Range, Regex, Enum, Dtype, NotNull, Unique, CrossColumn), auto-inferred from clean data, validated in grading
- **datasets/** — 25-entry dataset catalog plus download pipeline in `tools/download_datasets.py`
- **tools/benchmark_runner.py** — Config-driven benchmark orchestrator: dataset×category×difficulty×model×seed task matrix; saves results.jsonl + summary.csv + per-task JSONL episode logs to `outputs/benchmark/`; accepts `--model-name`/`--api-base` for single-model runs or `--models` to filter from config; retries up to 3× with backoff on CAPACITY_REACHED errors
- **tools/benchmark_config.yaml** — Benchmark config: 6 local GGUF models (Qwen3.5-0.8B/2B/9B, gemma-4-E2B/E4B, Qwen3-4B), 6 categories, 3 difficulties, all discovered datasets
- **run_benchmark.sh** — Multi-model benchmark orchestrator: loops through GGUF models, starts/stops llama-server per model, delegates to benchmark_runner; supports `--models`, `--categories`, `--difficulties` filters
- **run_all_models.sh** — Legacy multi-model runner (calls inference.py directly, no category tagging)
- **outputs/benchmark/** — Git-tracked benchmark results: `results.jsonl`, `summary.csv`, `episodes/*.jsonl`
- **ui/** — Gradio benchmark dashboard (dark theme): category-card leaderboard with bar charts, step-by-step episode explorer, dataset catalog with rule viewer

## Key Decisions

- **Code generation over structured transforms** — agents write real pandas code
- **Generative episodes over static tasks** — `reset()` samples a dataset/profile/format and creates corruptions dynamically; `LEGACY_TASK_MAP` preserves old task IDs for compatibility
- **Multi-format input** — agents may receive csv, json, jsonl, excel, tsv, xml, fixed-width, html table, or sql dump with file previews and diagnosis metadata in observations
- **Multi-level grading over simple diff scoring** — reward combines schema_score (0.15), row_score (0.15/0.20), cell_score (0.50/0.55), distribution_score (0.10), and semantic_score (0.10 when rules present)
- **Content-based row matching** — row recovery matched by content hash rather than index; `_row_hash` normalizes numeric values (`6.0 → "6"`) so CSV round-trip int/float variants match correctly; critical for drop_rows/duplicate_rows/reorder tasks
- **Wrong-value penalty** — changing a cell to an incorrect value is penalized 1.5×; numeric near-misses (≤5% relative error) get graduated partial credit instead of full penalty
- **Collateral damage penalty** — cells that were correct but got corrupted by the agent add 0.5 severity each; detection uses row_mapping so reordered rows are aligned before comparison (no phantom penalties)
- **Accepted null fill tolerance** — `accepted_fill` field per null error: "any" accepts reasonable imputation (mean, median, mode, ffill, bfill) within 0.5σ of column range; "exact" requires the clean value; "mean"/"median"/"mode" accept only that specific statistic
- **Explore cost** — each explore action incurs a small efficiency penalty (0.01/step, 0.03 for timeouts), discouraging excessive or wasteful exploration
- **Undo cost** — undo restores the last filesystem checkpoint and applies a configured score cost through `undo_count`
- **Validate budget** — agents get 2 validate actions per episode; returns structured per-column/per-corruption-type diagnosis without ending the episode; tracks `validate_count`/`validate_uses`
- **Soft done** — first done is a checkpoint if reward < 1.0: agent sees score + remaining errors and can continue. Second done is final. Perfect score always ends immediately. Capped at 1 retry
- **Persistent worker** — pandas/numpy loaded once per episode; transform uses `df.copy()` in exec namespace (no CSV re-read per step); explore uses in-memory `df` directly (read-only); undo sends `"reload"` command to resync worker state after checkpoint restore
- **Auto-rewrite inplace=True** — worker auto-converts `df['col'].fillna(val, inplace=True)` → `df['col'] = df['col'].fillna(val)` (pandas 2.x broke chained inplace)
- **Difficulty profiles** — easy stays focused (1 type, CSV only); medium uses 3-4 corruption families with capped column spread and light format noise; hard uses wide row/schema/format mix with row-level ops enabled
- **Format-aware corruption safety** — `CorruptionPipeline.select_format()` is called before `corrupt()` so raw dirty content, parser expectations, hints, and previews stay aligned
- **Inference config** — `inference.py` reads `API_BASE_URL`, `MODEL_NAME`, and `API_KEY` (from `HF_TOKEN`/`OPENAI_API_KEY`) at import time via `os.environ.get`; validator injects LiteLLM proxy values which take precedence
- **Inference prompting** — explicit warning that `pd.to_numeric(errors='coerce')` converts sentinels to NaN which grades as wrong_value (1.5× penalty); must fill NaN in same transform; warns about sentinel auto-parsing ("n/a", "NA", "null" → NaN by pd.read_csv before agent sees them)

## Conventions

- OpenEnv spec v1: typed models, `reset()`/`step()`/`state()` API
- Dual-import in server files: `try: from ..models / except: from models`
- Rewards in 0.0–1.0 range, with schema/row/cell/distribution components plus action costs
- `inference.py` LLM calls use `OpenAI(base_url=API_BASE_URL, api_key=API_KEY)` — env vars read once at import
- Client never imports server
- Sandbox always on for code execution
- Corruption subsystem owns task generation, format selection, hints, and raw dirty content
- Generator owns all domain knowledge — grader is a pure diff engine

## Invariants

- Rewards always in [0.0, 1.0]
- Sandbox always on — no code execution without AST scan + persistent worker
- Grader never knows about corruption implementations — scores result quality from clean data, error maps, and generated metadata
- Grader detects collateral damage using content-based row mapping so reordered rows are aligned before comparison
- Dynamic reset produces clean data, dirty raw content, normalized CSV working copy, error map, metadata, hints, and observation previews
- CSV is the source of truth between steps; worker in-memory `df` is kept in sync (reload on undo)
- `CorruptionPipeline.select_format()` must always be called before `corrupt()` — format selection affects which corruption types are applied
- Client never imports server
- Dual-import pattern in server files: `try: from ..models / except: from models`

## Commands

```bash
source .venv/bin/activate

# ── Environment Server ────────────────────────────────────────
# Run server (with WebSocket keepalive for long LLM calls)
uvicorn server.app:app --port 8000 --ws-ping-interval 60 --ws-ping-timeout 120

# ── Inference (single model, requires running env server + LLM server) ──
# Via wrapper script (has pre-flight checks for env server + LLM API)
./inference.sh                              # all 18 eval tasks
./inference.sh titanic/easy/csv             # single task
# Or directly
python inference.py
python inference.py titanic_easy wine_medium
# Environment variables for inference.sh:
#   API_BASE_URL   (default: http://localhost:8080/v1)
#   OPENAI_API_KEY (default: dummy)
#   MODEL_NAME     (default: gemma-4-E2B-it-Q4_K_M)
#   ENV_URL        (default: http://localhost:7860)
#   MIN_CALL_INTERVAL (default: 0)

# ── Benchmark (multi-model, auto-manages llama-server lifecycle) ──
# Full benchmark: all 6 models × all datasets × 3 difficulties
# Requires: env server running, all model GGUFs in ~/models/
# See docs/setup-llama-mac.md for model downloads
./run_benchmark.sh
# Filter by model/category/difficulty
./run_benchmark.sh --models "Qwen3.5-0.8B-UD-Q4_K_XL" --categories FP VR --difficulties easy medium
# Single model via benchmark_runner directly (llama-server must be running)
python -m tools.benchmark_runner --model-name "Qwen3.5-0.8B-UD-Q4_K_XL" --api-base http://localhost:8080/v1
# Benchmark models (6):
#   Qwen3.5-0.8B-UD-Q4_K_XL, Qwen3.5-2B-UD-Q4_K_XL, Qwen3.5-9B-UD-Q4_K_XL
#   gemma-4-E2B-it-Q4_K_M, gemma-4-E4B-it-Q4_K_M, Qwen3-4B-Q4_K_M
# Output: outputs/benchmark/{results.jsonl, summary.csv, episodes/*.jsonl}
# Resumable: skips already-completed (dataset, category, difficulty, model, seed) combos
# Sampling: randomly pick N new tasks instead of running full matrix
./run_benchmark.sh --max-tasks 50
# Or set max_tasks in tools/benchmark_config.yaml (CLI overrides YAML)

# ── Benchmark with paid APIs (OpenAI, Groq, etc.) ────────────
# --api-key-env is the NAME of the env var, not the key itself
export OPENAI_API_KEY="sk-..."
python -m tools.benchmark_runner --model-name "gpt-4o" --api-base "https://api.openai.com/v1" --api-key-env "OPENAI_API_KEY" --max-tasks 20
# Groq example:
export OPENAI_API_KEY="gsk-..."
python -m tools.benchmark_runner --model-name "llama-3.3-70b" --api-base "https://api.groq.com/openai/v1" --api-key-env "OPENAI_API_KEY" --max-tasks 20
# Note: set min_call_interval in benchmark_config.yaml to 2.5+ for rate-limited APIs

# ── UI (Gradio benchmark dashboard) ──────────────────────────
python -m ui.app                            # default port 7861
python -m ui.app --port 7862                # custom port
# Also available at /web when env server runs with ENABLE_WEB_INTERFACE=true

# ── Deployment (HF Spaces) ───────────────────────────────────
# Push to HF Space remote (includes benchmark results + episodes)
git push hf main
# Dockerfile runs env server with Gradio UI on port 7860
# README.md frontmatter controls Space settings (default_tab: custom)

# ── Data & Datasets ───────────────────────────────────────────
python tools/download_datasets.py           # download/update dataset catalog inputs
python tools/corruption/engine.py           # generate all task artifacts

# ── UI ─────────────────────────────────────────────────────────
# Launch Gradio dashboard (reads from outputs/benchmark/)
python -m ui.app
python -m ui.app --port 7862 --benchmark-dir outputs/benchmark
```

## Benchmark Status

The old model benchmark tables were captured before the April 2026 difficulty rebalance and are no longer comparable to the current generator. Do not use the old `titanic_medium`/`wine_medium` reward numbers as current baselines.

### Post-rebalance profile sanity check (Titanic, CSV, seeds 42-51)

| Difficulty | Typical total errors | Average total errors | Notes |
|------------|----------------------|----------------------|-------|
| easy       | 33-98                | 61.5                 | Single focused corruption type |
| medium     | 136-447              | 278.0                | 3-4 corruption types with capped column spread |
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
- ~~Grader index type mismatch~~ — fixed: content-based row matching (`match_rows_by_content`) now used throughout; `_row_hash` handles OverflowError
- ~~Grader `_row_hash` int/float mismatch~~ — fixed: CSV round-trip converts int columns to float when NaN is present, causing `str(6) ≠ str(6.0)` and zero row matches in `match_rows_by_content`. `_row_hash` now normalizes numeric values (`6.0 → "6"`)
- Medium/hard 0-progress with small models: not an environment bug — `pd.to_numeric(errors='coerce')` converts sentinels like `"##"` → NaN when clean value is `0`, graded as `wrong_value` (1.5× penalty). Models must impute correct values, not just coerce to NaN
- ~~Collateral damage detection could compare dropped rows against shifted result indices~~ — fixed: row-mapping aware, skips clean rows missing from the content mapping after row drops
- ~~Missing rows never graded as fixed~~ — fixed: error_map keys for missing rows use `"missing_"` prefix with clean indices; grader handles both spurious and missing rows via content matching
- ~~Hard difficulty dtype bug (RangeRule int64 crash)~~ — fixed: `business_rule_violation` now casts float bad_val to int for integer columns
- ~~Error map keys used dirty indices after drop_rows~~ — fixed: all cell corruption functions now use `_error_key()` helper that resolves clean row index via `_get_clean_index_map`
- ~~Explore/validate costs not reflected in observation reward~~ — fixed: `_handle_explore` now calls `_ensure_graded()` before returning observation; validate already did
- ~~Validate/explore counters not marking reward stale~~ — fixed: both handlers set `_reward_stale = True` before proceeding
- ~~Invalid task_id silently picked random dataset~~ — fixed: now raises `ValueError`
- ~~Schema score cached across column changes~~ — fixed: cache invalidated when column list changes
- ~~Cross-column rules not built from clean data~~ — fixed: built at reset time and passed to grade()
- ~~Explore timeout classification counted syntax errors~~ — fixed: only actual timeout/timed-out errors count
- ~~Benchmark runner no retry on capacity errors~~ — fixed: up to 3 retries with backoff for CAPACITY_REACHED
- ~~Grader row_mapping fallback after drop_duplicates caused phantom wrong_value penalties~~ — fixed: unmapped rows marked "unfixed" instead of falling back to same index (which pointed to different row after row operations)
- ~~`_row_hash` str/float mismatch in FP tasks~~ — fixed: string guard prevented normalizing `"0.0"` (str) to match `0.0` (float), causing `match_rows_by_content` to return 0 matches when dirty data had string-typed columns. All FP/format tasks got cell_score=0.0 permanently. Removed the string guard; any value that parses as float normalizes uniformly.
- Pre-existing: `test_apply_format_corruptions_difficulty[medium]` fails — `apply_format_corruptions` returns empty list for medium CSV
- wine_quality FP easy: `drop_duplicates()` over-removes rows (1658→1360 vs 1599 clean) because the dataset has 240 natural duplicates. Inference strategy issue, not grading bug.

## File Map

- `models.py` — Pydantic action, observation, state, error-map, undo, and validate types
- `server/environment.py` — Generative env loop, `LEGACY_TASK_MAP`, action dispatch, observation building, soft-done logic
- `server/sandbox.py` — Persistent worker management, raw file/CSV setup, AST safety (exec + eval modes), filesystem checkpoints, env stripping, atexit cleanup
- `server/worker.py` — Worker process: exec/eval agent code with restricted `__builtins__`, inplace rewriting, `_BoundedStringIO` stdout cap, 2GB memory limit (Linux), reload-on-undo
- `server/grader.py` — Multi-level reward formula, content-based row matching with numeric normalization, collateral damage via row_mapping, semantic scoring, validation diagnostics
- `server/app.py` — FastAPI wiring, optional Gradio at `/web`
- `client.py` — WebSocket client
- `inference.py` — LLM agent baseline (model-agnostic), 18-task eval suite, auto-transforms (duplicate_rows, inject_nulls median/mode, whitespace_noise), auto-undo on regression with best_reward preservation, post-undo warnings, NaN coerce warnings in prompt
- `server/corruption/pipeline.py` — Runtime `CorruptionPipeline`, format selection (`select_format()` before `corrupt()`), corruption orchestration
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
- `run_benchmark.sh` — Multi-model benchmark shell orchestrator: llama-server lifecycle per model, delegates to benchmark_runner
- `run_all_models.sh` — Legacy multi-model runner (calls inference.py directly)
- `tools/benchmark_runner.py` — Config-driven benchmark orchestrator: task matrix generation, per-task JSONL episode capture, results.jsonl + summary.csv output
- `tools/benchmark_config.yaml` — Benchmark config: 6 models, 6 categories, 3 difficulties
- `ui/app.py` — Gradio dashboard entry point, injects dark CSS
- `ui/theme.py` — Dark CSS, category metadata (FP/VR/MD/SR/SV/CP), model color palette
- `ui/leaderboard.py` — Category-card grid with expandable bar charts per category
- `ui/explorer.py` — Step-by-step episode replay from JSONL logs
- `ui/catalog_view.py` — Dataset catalog browser with semantic rule viewer
- `ui/data_loader.py` — Loads benchmark results (results.jsonl primary, results.csv fallback), JSONL episodes, catalog; infers category when not present

## Task IDs

- `inference.py` defines an 18-task eval suite in `EVAL_TASKS` — pinned (dataset, difficulty, format) triples, 3 per dataset across 6 datasets
- Legacy IDs such as `titanic_easy`, `titanic_medium`, `titanic_hard`, `wine_easy`, `wine_medium`, and `wine_hard` are mapped through `LEGACY_TASK_MAP` in `environment.py` for backward compatibility

## Eval Task Suite (18 tasks — 3 per dataset)

| Dataset | Easy | Medium | Hard |
|---------|------|--------|------|
| Titanic | csv | csv | csv |
| Iris | csv | csv, jsonl | — |
| Boston Housing | — | csv | csv, json |
| Diabetes | — | csv | csv, json |
| Wine Quality | csv | csv | csv |
| Breast Cancer | csv | csv, jsonl | — |

## Grading Formula

```
# With semantic rules (auto-inferred from clean data):
base_score =
  schema_score       × 0.15
  row_score          × 0.15
  cell_score         × 0.50
  distribution_score × 0.10
  semantic_score     × 0.10

# Without rules (legacy):
base_score =
  schema_score       × 0.15
  row_score          × 0.20
  cell_score         × 0.55
  distribution_score × 0.10

cell_score accounts for exact fixes, accepted null fills (0.5σ margin), numeric near-misses (≤5% relative error),
wrong-value penalties (1.5×), whitespace/format tolerance, and collateral damage (0.5 severity per cell).
row_score uses content-based matching with numeric normalization for dropped/duplicated/reordered rows.
semantic_score measures rule violation rate against 7 auto-inferred constraint types.

transform_penalty = max(0, transform_steps - min_steps) / (max_steps × 2)
explore_penalty   = (successful_explores × 0.01) + (timed_out_explores × 0.03)
undo_penalty      = undo_count × undo_cost
validate_penalty  = validate_count × validate_cost
efficiency_factor = max(0.5, 1.0 - transform_penalty - explore_penalty - undo_penalty - validate_penalty)
reward = base_score × efficiency_factor   [clamped to 0.0–1.0]
```

## Environment Variables

| Var | Default | Purpose |
|-----|---------|---------|
| `API_BASE_URL` | `https://api.openai.com/v1` | LLM endpoint; validator injects LiteLLM proxy here |
| `API_KEY` | from `HF_TOKEN` or `OPENAI_API_KEY` | API token (read at import time) |
| `MODEL_NAME` | `gpt-4o` | Model name |
| `ENV_URL` | `http://localhost:7860` | OpenEnv server URL |
| `LOG_LEVEL` | `INFO` | `INFO` for actions/timing, `DEBUG` for full LLM I/O |
| `LOG_DIR` | `outputs/logs` | JSONL log directory |
| `MIN_CALL_INTERVAL` | `2.5` | Min seconds between LLM calls (0 for local) |
| `TASKS_DIR` | `tasks` | Task config directory |
| `DATA_DIR` | `data` | Data artifacts directory |
| `DATASETS_DIR` | `datasets` | Dataset catalog/materialized inputs |
| `SANDBOX_BASE` | `outputs/sandbox` | Sandbox working directories |
| `ENABLE_WEB_INTERFACE` | unset | Set to enable Gradio dashboard at `/web` |

## Deployment Notes

- Validator/deployed runs should rely on the injected LiteLLM proxy values when they are present.
- The current script defaults `API_BASE_URL` to `https://api.openai.com/v1` and `MODEL_NAME` to `gpt-4o` when unset.
- `API_KEY` is read from `HF_TOKEN` then `OPENAI_API_KEY` at import time (empty string if neither set).
