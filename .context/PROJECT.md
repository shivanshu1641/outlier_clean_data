# Data Cleaning OpenEnv Environment — Project Context

## Purpose
OpenEnv environment for the Meta PyTorch Hackathon (deadline: April 8, 2026). AI agents clean dirty CSV data by writing Python/pandas code, graded on severity-weighted constraint satisfaction and step efficiency.

## Architecture
- **server/environment.py** — Core Environment class (reset/step/state)
- **server/sandbox.py** — Subprocess-based code execution with AST safety scanning
- **server/grader.py** — Constraint checkers + severity-weighted scoring
- **server/app.py** — FastAPI via `openenv.core.create_app()`
- **models.py** — Pydantic types: ExploreAction, TransformAction, DoneAction, Observation, State
- **client.py** — EnvClient subclass (WebSocket)
- **inference.py** — Baseline agent using OpenAI Client with [START]/[STEP]/[END] logging

## Key Decisions
- **Code generation over structured transforms** — more impressive, realistic, agents write real pandas code
- **Constraint satisfaction over ground-truth diff** — avoids multiple-valid-solutions problem
- **Subprocess sandbox over exec/RestrictedPython** — process isolation, timeout, memory limits
- **Severity-weighted scoring** — critical=3x, high=2x, medium=1x weight

## Invariants
- Rewards always in [0.0, 1.0]
- Sandbox always on — no code execution without AST scan + subprocess
- Constraint-based grading only — no ground-truth comparison
- Client never imports server
- Dual-import pattern in server files: `try: from ..models / except: from models`

## Commands
```bash
source .venv/bin/activate
uvicorn server.app:app --port 8000          # Run server
python inference.py                          # Run agent
python tools/corruption/engine.py            # Regenerate dirty data
```

## File Map
- `models.py` — All Pydantic types
- `server/environment.py` — Core env loop
- `server/sandbox.py` — Safe code execution
- `server/grader.py` — 10 constraint checkers + scoring
- `server/app.py` — FastAPI wiring
- `client.py` — WebSocket client
- `inference.py` — LLM agent baseline
- `tasks/*.json` — Task configs with constraints and severity
- `data/clean/` — Source datasets (Titanic, Wine Quality)
- `data/dirty/` — Corrupted datasets
- `tools/corruption/engine.py` — Standalone corruption generator
