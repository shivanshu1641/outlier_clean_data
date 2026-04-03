# Data Cleaning OpenEnv Environment

Read `.context/PROJECT.md` for full project context, architecture, conventions, and invariants.

## Quick Reference

```bash
source .venv/bin/activate
# Run server locally (with WebSocket keepalive for long LLM calls)
uvicorn server.app:app --port 8000 --ws-ping-interval 60 --ws-ping-timeout 120
# Run inference
python inference.py
# Run specific tasks
python inference.py titanic_easy wine_medium
# Generate all task artifacts (clean, dirty, error_map, severity_map)
python tools/corruption/engine.py
```

## Conventions
- OpenEnv spec v1: typed models, `reset()`/`step()`/`state()` API
- Dual-import in server files: `try: from ..models / except: from models`
- Rewards in 0.0–1.0 range, diff-based grading only
- Client never imports server
- Sandbox always on for code execution
- Generator owns all domain knowledge — grader is a pure diff engine
