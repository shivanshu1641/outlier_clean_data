# Runbook: Run Server

## Command
```bash
source .venv/bin/activate
uvicorn server.app:app --port 8000 --ws-ping-interval 60 --ws-ping-timeout 120
```

## Why the WebSocket flags
LLM API calls can take 20-30s. Without extended keepalive, the WebSocket drops with:
```
ConnectionClosedError: received 1011 (internal error) keepalive ping timeout
```
`--ws-ping-interval 60` and `--ws-ping-timeout 120` give enough headroom for slow model responses.

## Environment variables
All read from `.env` (loaded by the server process):

| Var | Default | Effect |
|-----|---------|--------|
| `TASKS_DIR` | `tasks` | Where task JSON configs are loaded from |
| `DATA_DIR` | `data` | Root for data artifacts |
| `SANDBOX_BASE` | `outputs/sandbox` | Where per-episode sandboxes are created |

## Health check
```bash
curl http://localhost:8000/health
# or
curl http://localhost:8000/info
```

## Prerequisites
- Run `python tools/corruption/engine.py` first to generate task artifacts
- `.venv` must be activated (or use `uvicorn` from the venv directly)
