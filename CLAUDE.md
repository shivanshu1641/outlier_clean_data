# Data Cleaning OpenEnv Environment

Read `.context/PROJECT.md` for full project context, architecture, conventions, and invariants.

## Quick Reference

```bash
source .venv/bin/activate
# Run server locally
uvicorn server.app:app --port 8000
# Run inference
python inference.py
# Generate dirty data (standalone tool)
python tools/corruption/engine.py
```

## Conventions
- OpenEnv spec v1: typed models, `reset()`/`step()`/`state()` API
- Dual-import in server files: `try: from ..models / except: from models`
- Rewards in 0.0–1.0 range, constraint-based grading only
- Client never imports server
- Sandbox always on for code execution
