# Architecture

## Overview

A data cleaning RL environment where an AI agent receives dirty CSV data and must fix it using Python/pandas code, evaluated against the ground truth.

## System Diagram

```
 OFFLINE (one-time)                    RUNTIME (per episode)
+----------------------+    +--------------------------------------------+
|                      |    |                                            |
| tools/corruption/    |    |  inference.py            server/app.py     |
| engine.py            |    |  +------------+   WS    +----------+       |
|      |               |    |  |    LLM     |<------->|  FastAPI |       |
|      v               |    |  |    Agent   |         +----+-----+       |
| +----------+         |    |  +-----+------+              |             |
| | clean    |         |    |        | action               v            |
| | .csv     |---------+--->|        |              +--------------+     |
| +----------+  loaded |    |        |              | environment  |     |
| | dirty    |  at     |    |        |              | .py          |     |
| | .csv     |  reset  |    |        |              +------+-------+     |
| +----------+         |    |        |                     |             |
| | error_map|---------|----+---->   |              +------+-------+     |
| | .json    |  used   |    |        |              |      |       |     |
| +----------+  for    |    |        |          +---+---+  +---+---+     |
| | severity |  grading|    |        |          |grader |  |sandbox|     |
| | _map.json|         |    |        |          |.py    |  |.py    |     |
| +----------+         |    |        |          +-------+  +---+---+     |
|                      |    |        |                         |         |
+----------------------+    |        |                    +----+----+    |
                            |        |                    | worker  |    |
                            |        |                    | .py     |    |
                            |        |                    +----+----+    |
                            |        |                         |         |
                            |        v                         v         |
                            |     action                  current.csv    |
                            |     {explore,transform,done} (sandboxed)   |
                            +---------------------------------------------+
```

## Agent Loop

```
                  +----------+
                  |  reset   |
                  +----+-----+
                       |
                       v
                +-----------+
           +--->|observation|
           |    +-----+-----+
           |          |
           |          v
           |    +-----------+
           |    | LLM call  |
           |    +-----+-----+
           |          |
           |    +-----+-----+--------+
           |    |           |        |
           |    v           v        v
           | explore   transform   done
           | (cheap)   (costly)      |
           |    |           |    +---+---+
           |    |           |    | grade |
           |    |           |    +---+---+
           |    |           |        |
           |    |           |   score < 1.0?
           |    |           |    yes/    \no
           |    |           |   /         \
           |    v           v  v     +--------+
           | +--------+  soft done   | final  |
           +-+ step   |  (retry 1x)  | done   |
             +--------+              +--------+
```

## Data Flow

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
  DataCleaningEnvironment.reset()
    -> load error_map.json + clean.csv
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
  final grade -> return Observation (done=True)

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

## Grading

The grader receives:

- `clean_df` -- ground truth
- `result_df` -- agent's current DataFrame
- `error_map` -- pre-built map of all corrupted cells/rows (from generator)

For each error, it checks the result:

- **Fixed** (result == clean value) -> 0 severity remaining
- **Unfixed** (still dirty) -> full severity remaining
- **Wrong value** (changed, but not to clean value) -> 1.5x severity remaining

```
constraint_score  = 1 - (remaining_severity / total_severity)

transform_penalty = max(0, transform_steps - min_steps) / (max_steps * 2)
explore_penalty   = (successful_explores * 0.01) + (timed_out_explores * 0.03)
efficiency_factor = max(0.5, 1.0 - transform_penalty - explore_penalty)

reward = constraint_score * efficiency_factor   [clamped to 0.0-1.0]
```

Explores are cheap but not free. Timed-out explores cost 3x a successful one. The 0.5 floor ensures even heavy explorers get half credit for good cleaning.

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
