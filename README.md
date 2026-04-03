# Data Cleaning OpenEnv Environment

An [OpenEnv](https://github.com/meta-pytorch/OpenEnv) environment where AI agents learn to clean dirty CSV data by writing Python/pandas code, graded on severity-weighted constraint satisfaction and step efficiency.

Built for the [Meta PyTorch OpenEnv Hackathon x Scaler School of Technology](https://www.scaler.com/school-of-technology/meta-pytorch-hackathon/).

## Quick Start

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -r server/requirements.txt

# Start the environment server
uvicorn server.app:app --port 8000 --ws-ping-interval 60 --ws-ping-timeout 120

# Run the baseline agent (in another terminal)
API_BASE_URL=http://localhost:11434/v1 MODEL_NAME=qwen3 ENV_URL=http://localhost:8000 python inference.py

# Run specific tasks
python inference.py titanic_easy wine_medium
```

## Architecture

```
[Generator]  tools/corruption/engine.py
  load_titanic() / load_wine()
       |
  apply_corruptions(df, config, error_log)
       |
  4 artifacts per task:
    clean.csv          <-- ground truth
    dirty.csv          <-- agent input
    error_map.json     <-- cell/row errors (what, where, clean value, severity)
    severity_map.json  <-- totals per corruption type

[Server]  server/app.py -> environment.py -> grader.py
  POST /reset?task_id=titanic_easy
       |
  Environment.reset()
    -> load error_map + clean.csv
    -> create sandbox (dirty.csv -> current.csv)
    -> grade(clean_df, dirty_df, error_map) -> initial reward
    -> return Observation

  POST /step  {type: "explore", query: "df.head()"}
       |
  execute_explore(query, worker)  <-- subprocess, read-only
    -> return Observation (no state change, small efficiency cost)

  POST /step  {type: "transform", code: "df['Age'].fillna(...)"}
       |
  execute_transform(code, worker)  <-- subprocess, AST-checked
    -> overwrite current.csv
    -> grade(clean_df, result_df, error_map) -> updated reward
    -> return Observation

  POST /step  {type: "done"}
       |
  if first done AND score < 1.0:
    -> soft done: return score + remaining errors (done=False, agent continues)
  else:
    -> hard done: final grade, return Observation (done=True)

[Client]  client.py
  DataCleaningClient(base_url=ENV_URL)
    .reset(task_id=...) -> StepResult
    .step(action)       -> StepResult

[Inference]  inference.py
  for task in task_ids:
    obs = env.reset(task_id)
    while not done:
      action = llm(system_prompt, obs)
      obs = env.step(action)
```

## Component Responsibilities

| Component | Owns |
|-----------|------|
| `tools/corruption/engine.py` | All domain knowledge: what corruptions exist, their severity, how to track them |
| `server/grader.py` | Pure diff engine: compare result vs clean using error_map, compute reward |
| `server/environment.py` | Episode lifecycle, sandbox management, observation building |
| `server/sandbox.py` | Code safety (AST scan), subprocess isolation, timeout/memory limits |
| `server/worker.py` | Worker process: exec/eval agent code, auto-rewrite `inplace=True` patterns |
| `inference.py` | LLM orchestration, prompt building, action parsing |

## How It Works

1. **Reset**: Agent receives a dirty dataset + error summary
2. **Explore**: Agent inspects the data (pandas queries, read-only, 10/cycle budget, small cost per call)
3. **Transform**: Agent submits Python/pandas code executed in a sandbox
4. **Grade**: Diff-based grading against ground truth, severity-weighted
5. **Done**: Agent signals completion — if score < 1.0 on first done, agent gets a second chance (soft done). Second done or perfect score ends the episode

## Action Space

| Action | Description |
|--------|-------------|
| `ExploreAction(query="df.isnull().sum()")` | Read-only data inspection |
| `TransformAction(code="df['Age'] = df['Age'].fillna(0)")` | Execute pandas code in sandbox |
| `DoneAction()` | End the episode |

## Grading

The grader compares the agent's result to the ground truth using a pre-built error map:

- **Fixed** (result == clean value) -> 0 severity remaining
- **Unfixed** (still dirty) -> full severity remaining
- **Wrong value** (changed to wrong value) -> 1.5x severity remaining

```
constraint_score  = 1 - (remaining_severity / total_severity)

transform_penalty = max(0, transform_steps - min_steps) / (max_steps * 2)
explore_penalty   = (successful_explores * 0.01) + (timed_out_explores * 0.03)
efficiency_factor = max(0.5, 1.0 - transform_penalty - explore_penalty)

reward = constraint_score * efficiency_factor   [clamped to 0.0-1.0]
```

Explores are cheap but not free. Timed-out explores cost 3x a successful one. The 0.5 floor ensures even heavy explorers get half credit for good cleaning.

## Tasks

| Task | Dataset | Errors | Difficulty |
|------|---------|--------|------------|
| `titanic_easy` | Titanic (891 rows) | 59 | Easy -- nulls, whitespace |
| `titanic_medium` | Titanic (891 rows) | 277 | Medium -- + type errors |
| `titanic_hard` | Titanic (891 rows) | 958 | Hard -- + outliers, dupes, format |
| `wine_easy` | Wine Quality (1599 rows) | 255 | Easy -- nulls |
| `wine_medium` | Wine Quality (1599 rows) | 836 | Medium -- + dupes, types, outliers |
| `wine_hard` | Wine Quality (1599 rows) | 2312 | Hard -- + heavy nulls, whitespace |

## Sandbox Isolation

Each episode gets an isolated directory:

```
outputs/sandbox/{episode_id}/
  input.csv       (original dirty data, never modified)
  current.csv     (working copy, updated after each transform)
  scripts/        (generated transform scripts for replay)
  artifacts/      (snapshots after each transform)
```

Agent code runs in a subprocess with:
- AST-level import/call blocking (no os, sys, subprocess, open, eval, etc.)
- 30s transform timeout, 10s explore timeout
- 2GB memory limit

## Benchmark Results

### Gemma 4 E2B (2B params, local, 8K context)

| Task | Errors Fixed | Steps | Reward | Notes |
|------|-------------|-------|--------|-------|
| titanic_easy | 59/59 | 2 | 0.99 | Explore penalty: -0.01 |
| titanic_medium | 233/277 | 6 | 0.66 | Soft done triggered, context overflow on retry |
| titanic_hard | 764/958 | 7 | 0.64 | Soft done triggered, context overflow on retry |
| wine_easy | 255/255 | 2 | 0.99 | Explore penalty: -0.01 |
| wine_medium | 591/836 | 6 | 0.50 | Soft done triggered, context overflow on retry |
| wine_hard | 1418/2312 | 8 | 0.48 | Soft done triggered, context overflow on retry |
| **Average** | | | **0.71** | |

### GPT-4.1 Nano (OpenAI API, with action history + loop detection)

| Task | Errors Fixed | Steps | Reward | Notes |
|------|-------------|-------|--------|-------|
| titanic_easy | 59/59 | 4 | 0.97 | 3 explores, auto-done on all fixed |
| titanic_medium | 233/277 | 5 | 0.66 | Soft done triggered, second done finalizes |
| titanic_hard | 764/958 | 9 | 0.63 | Auto-done after 3 stale transforms at 764 fixed |
| wine_easy | 255/255 | 4 | 0.90 | 2 explores, auto-done on all fixed |
| wine_medium | 566/836 | 13 | 0.40 | Heavy explores (10), soft done + second done |
| wine_hard | 22/2312 | 7 | 0.00 | Model gives up early — complexity beyond Nano's capability |
| **Average** | | | **0.53** | |

### Key Observations

- **Gemma 4 E2B (2B, local)** outperforms **GPT-4.1 Nano** despite being much smaller — likely because Gemma writes more comprehensive transforms per step
- **Action history + loop detection** prevents the worst failure modes: titanic_medium improved from 0.37 to 0.66 (no more repeated transforms), titanic_hard auto-dones at peak instead of spiraling down
- **Auto-done circuit breakers** work: stale detection (3 transforms, same errors fixed) and regression detection (2 consecutive transforms reducing errors) prevent wasted steps
- **Soft done** works correctly but the 2B local model hits context limits on retry; larger-context models would benefit more
- **Explore penalty** is visible: titanic_easy scores 0.97 (not 1.0) due to explore steps costing 0.01 each

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `API_BASE_URL` | `http://localhost:11434/v1` | LLM endpoint (any OpenAI-compatible) |
| `API_KEY` | `` | API key (empty for local) |
| `MODEL_NAME` | `qwen3` | Model name |
| `ENV_URL` | `http://localhost:8000` | OpenEnv server URL |
| `LOG_LEVEL` | `INFO` | `INFO` for actions/timing, `DEBUG` for full LLM I/O |
| `LOG_DIR` | `outputs/logs` | JSONL log directory |
| `MIN_CALL_INTERVAL` | `2.5` | Min seconds between LLM calls (0 for local) |

## Project Structure

```
models.py                  # Typed Pydantic models (Action, Observation, State)
client.py                  # EnvClient subclass (WebSocket)
inference.py               # Baseline LLM agent
server/
  environment.py           # Core reset/step/state loop
  sandbox.py               # Sandboxed code execution + AST safety
  worker.py                # Worker process (exec/eval, inplace rewriting)
  grader.py                # Diff engine + reward formula
  app.py                   # FastAPI application
tasks/                     # Task configs (per difficulty)
data/                      # Clean + dirty datasets + error maps
tools/corruption/          # Standalone dirty data generator
docs/                      # Architecture, ADRs, runbooks
```

## Documentation

- [architecture.md](architecture.md) -- Architecture and data flow
