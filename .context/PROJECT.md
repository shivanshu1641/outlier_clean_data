# Data Cleaning OpenEnv Environment — Project Context

## Purpose

OpenEnv environment for the Meta PyTorch Hackathon (deadline: April 8, 2026). AI agents clean dirty CSV data by writing Python/pandas code, graded on how many errors they fix relative to the ground truth, with a step efficiency bonus.

## Architecture

- **server/environment.py** — Core Environment class (reset/step/state), detects no-op transforms
- **server/sandbox.py** — Persistent worker process per episode (pandas stays loaded in memory)
- **server/worker.py** — Worker process: re-reads CSV each step, auto-rewrites `inplace=True` patterns
- **server/grader.py** — Generic diff-based grader: compares result vs clean data using error map
- **server/app.py** — FastAPI via `openenv.core.create_app()`
- **models.py** — Pydantic types: ExploreAction, TransformAction, DoneAction, Observation, State
- **client.py** — EnvClient subclass (WebSocket)
- **inference.py** — Baseline agent using any OpenAI-compatible API with structured logging + JSONL persistence
- **tools/corruption/engine.py** — Standalone generator: produces 4 artifacts per task

## Key Decisions

- **Code generation over structured transforms** — agents write real pandas code
- **Diff-based grading over constraint checkers** — compares result to ground truth using a pre-built error map; generator owns all domain knowledge, grader is a pure diff engine
- **Wrong-value penalty** — changing a cell to an incorrect value is penalized 1.5× more than leaving it dirty
- **Explore cost** — each explore action incurs a small efficiency penalty (0.01/step, 0.03 for timeouts), discouraging excessive or wasteful exploration
- **Soft done** — first done is a checkpoint if reward < 1.0: agent sees score + remaining errors and can continue. Second done is final. Perfect score always ends immediately. Capped at 1 retry
- **Null fill tolerance** — `accepted_fill` field per null error: "any" accepts any reasonable imputation (mean, median, mode, ffill, bfill), "exact" requires the clean value, "mean"/"median"/"mode" accept only that specific statistic
- **Persistent worker** — pandas/numpy loaded once per episode, CSV re-read each step (avoids cold-start AND the pandas Copy-on-Write `inplace=True` pitfall)
- **Auto-rewrite inplace=True** — worker auto-converts `df['col'].fillna(val, inplace=True)` → `df['col'] = df['col'].fillna(val)` (pandas 2.x broke chained inplace)
- **Multi-difficulty per dataset** — titanic and wine each have easy/medium/hard variants

## Conventions

- OpenEnv spec v1: typed models, `reset()`/`step()`/`state()` API
- Dual-import in server files: `try: from ..models / except: from models`
- Rewards in 0.0–1.0 range, diff-based grading only
- Client never imports server
- Sandbox always on for code execution
- Generator owns all domain knowledge — grader is a pure diff engine

## Invariants

- Rewards always in [0.0, 1.0]
- Sandbox always on — no code execution without AST scan + persistent worker
- Grader never knows about corruption types — only compares values
- Generator always produces 4 artifacts + task config: clean.csv, dirty.csv, error*map.json, severity_map.json, task*\*.json
- CSV is the source of truth between steps, not in-memory df
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
# Regenerate all task artifacts
python tools/corruption/engine.py
```

## Verified Results (Gemma 4 E2B, 2B params, local)

| Task           | Errors Fixed | Reward   |
| -------------- | ------------ | -------- |
| titanic_easy   | 59/59        | 1.00     |
| titanic_medium | 233/277      | 0.67     |
| titanic_hard   | 764/958      | 0.66     |
| wine_easy      | 255/255      | 1.00     |
| wine_medium    | 611/836      | 0.52     |
| wine_hard      | 1418/2312    | 0.49     |
| **Average**    |              | **0.64** |

## Verified Results (GPT-4.1 Nano, OpenAI API, with action history + loop detection)

| Task           | Errors Fixed | Reward   |
| -------------- | ------------ | -------- |
| titanic_easy   | 59/59        | 0.97     |
| titanic_medium | 233/277      | 0.66     |
| titanic_hard   | 764/958      | 0.63     |
| wine_easy      | 255/255      | 0.90     |
| wine_medium    | 566/836      | 0.40     |
| wine_hard      | 22/2312      | 0.00     |
| **Average**    |              | **0.53** |

## Known Issues

- Small models (2B) give up early on hard tasks — larger models (70B+ via Groq) should do better
- Context window overflow on hard tasks with small local models (8192 tokens) — inference auto-submits `done` on context exceeded
- `titanic_hard` shows 16/958 fixed at reset — phantom matches from overlapping corruptions where dirty value happens to equal clean value

## File Map

- `models.py` — All Pydantic types
- `server/environment.py` — Core env loop
- `server/sandbox.py` — Persistent worker management + AST safety
- `server/worker.py` — Worker process (exec/eval agent code, inplace rewriting)
- `server/grader.py` — Diff engine + reward formula
- `server/app.py` — FastAPI wiring
- `client.py` — WebSocket client
- `inference.py` — LLM agent baseline (model-agnostic)
- `tasks/task_<task_id>.json` — Task configs (generated)
- `data/<task_id>/clean.csv` — Ground truth
- `data/<task_id>/dirty.csv` — Corrupted input
- `data/<task_id>/error_map.json` — Cell/row errors with severity, clean values, accepted_fill
- `data/<task_id>/severity_map.json` — Severity totals per corruption type
- `tools/corruption/engine.py` — Standalone corruption generator (also writes explore cost configs)
- `architecture.md` — System diagram, agent loop, data flow, grading details

## Task IDs

- `titanic_easy`, `titanic_medium`, `titanic_hard`
- `wine_easy`, `wine_medium`, `wine_hard`

## Grading Formula

```
constraint_score = 1 - (remaining_severity / total_severity)
  where:
    fixed cell (exact match or accepted fill) → 0 severity remaining
    unchanged dirty cell                      → full severity remaining
    wrong value cell                          → severity × 1.5 remaining
    whitespace/format cell (stripped match)    → 0 severity (fixed)
    spurious row still present                → full severity remaining

transform_penalty = max(0, transform_steps - min_steps) / (max_steps × 2)
explore_penalty   = (successful_explores × 0.01) + (timed_out_explores × 0.03)
efficiency_factor = max(0.5, 1.0 - transform_penalty - explore_penalty)
reward = constraint_score × efficiency_factor   [clamped to 0.0–1.0]
```

## Environment Variables

| Var                 | Default                     | Purpose                                             |
| ------------------- | --------------------------- | --------------------------------------------------- |
| `API_BASE_URL`      | `http://localhost:11434/v1` | LLM endpoint (any OpenAI-compatible)                |
| `API_KEY`           | ``                          | API key (empty = not-needed for local)              |
| `MODEL_NAME`        | `qwen3`                     | Model name                                          |
| `ENV_URL`           | `http://localhost:8000`     | OpenEnv server URL                                  |
| `LOG_LEVEL`         | `INFO`                      | `INFO` for actions/timing, `DEBUG` for full LLM I/O |
| `LOG_DIR`           | `outputs/logs`              | JSONL log directory                                 |
| `MIN_CALL_INTERVAL` | `2.5`                       | Min seconds between LLM calls (0 for local)         |
| `TASKS_DIR`         | `tasks`                     | Task config directory                               |
| `DATA_DIR`          | `data`                      | Data artifacts directory                            |
| `SANDBOX_BASE`      | `outputs/sandbox`           | Sandbox working directories                         |
