---
title: Data Cleaning OpenEnv
emoji: 🧹
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# Data Cleaning OpenEnv Environment

An [OpenEnv](https://github.com/meta-pytorch/OpenEnv) environment where AI agents learn to clean dirty CSV data by writing Python/pandas code, graded on severity-weighted constraint satisfaction and step efficiency.

Built for the [Meta PyTorch OpenEnv Hackathon x Scaler School of Technology](https://www.scaler.com/school-of-technology/meta-pytorch-hackathon/).

## Quick Start

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -r server/requirements.txt

# Start the environment server locally
uvicorn server.app:app --port 7860 --ws-ping-interval 60 --ws-ping-timeout 120

# Or run the submission Docker image from the repo root
docker build -t outlier-clean-data .
docker run -p 7860:7860 outlier-clean-data

# Run the baseline agent (in another terminal)
bash inference.sh

# Run specific tasks
bash inference.sh titanic easy json
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

| Component                    | Owns                                                                            |
| ---------------------------- | ------------------------------------------------------------------------------- |
| `tools/corruption/engine.py` | All domain knowledge: what corruptions exist, their severity, how to track them |
| `server/grader.py`           | Pure diff engine: compare result vs clean using error_map, compute reward       |
| `server/environment.py`      | Episode lifecycle, sandbox management, observation building                     |
| `server/sandbox.py`          | Code safety (AST scan), subprocess isolation, timeout/memory limits             |
| `server/worker.py`           | Worker process: exec/eval agent code, auto-rewrite `inplace=True` patterns      |
| `inference.py`               | LLM orchestration, prompt building, action parsing                              |

## How It Works

1. **Reset**: Agent receives a dirty dataset + error summary
2. **Explore**: Agent inspects the data (pandas queries, read-only, 10/cycle budget, small cost per call)
3. **Transform**: Agent submits Python/pandas code executed in a sandbox
4. **Grade**: Diff-based grading against ground truth, severity-weighted
5. **Done**: Agent signals completion — if score < 1.0 on first done, agent gets a second chance (soft done). Second done or perfect score ends the episode

## Action Space

| Action                                                    | Description                    |
| --------------------------------------------------------- | ------------------------------ |
| `ExploreAction(query="df.isnull().sum()")`                | Read-only data inspection      |
| `TransformAction(code="df['Age'] = df['Age'].fillna(0)")` | Execute pandas code in sandbox |
| `DoneAction()`                                            | End the episode                |

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

| Task family      | Dataset                  | Error count | Difficulty                         |
| ---------------- | ------------------------ | ----------- | ---------------------------------- |
| `*_easy`         | Any catalog dataset      | Dynamic     | Focused cleanup, csv-only, light corruption |
| `*_medium`       | Any catalog dataset      | Dynamic     | 3-4 corruption types, capped columns, light format noise |
| `*_hard`         | Any catalog dataset      | Dynamic     | Wide corruption mix, row-level ops, heavier format noise |

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

## Benchmark Status

The old benchmark tables were measured before the April 2026 difficulty rebalance and are no longer current. The generator now uses a more gradual curve, especially for `medium`, so those earlier reward numbers should be treated as historical only.

### Post-rebalance profile sanity check (Titanic, CSV, seeds 42-51)

| Difficulty | Typical total errors | Average total errors | Notes |
| ---------- | -------------------- | -------------------- | ----- |
| easy       | 33-98                | 61.5                 | Single focused corruption type |
| medium     | 136-447              | 278.0                | 3-4 corruption types with capped column spread |
| hard       | 1586-2895            | 2174.0               | Successful seeds only; some hard seeds still hit the dtype bug |

Full model reward tables should be rerun against the rebalanced generator before publishing new baselines.

## Environment Variables

| Variable            | Default                     | Purpose                                             |
| ------------------- | --------------------------- | --------------------------------------------------- |
| `API_BASE_URL`      | `https://router.huggingface.co/v1` | LLM endpoint; env value takes precedence       |
| `OPENAI_API_KEY` / `HF_TOKEN` | none             | API token env var used by `inference.py`            |
| `MODEL_NAME`        | `Qwen/Qwen2.5-72B-Instruct` | Model name                                          |
| `ENV_URL`           | `http://localhost:7860`     | OpenEnv server URL                                  |
| `LOG_LEVEL`         | `INFO`                      | `INFO` for actions/timing, `DEBUG` for full LLM I/O |
| `LOG_DIR`           | `outputs/logs`              | JSONL log directory                                 |
| `MIN_CALL_INTERVAL` | `2.5`                       | Min seconds between LLM calls (0 for local)         |

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
