# Runbook: Run Server

## Command
```bash
source .venv/bin/activate
python server/app.py
# or
uvicorn server.app:app --port 7860 --ws-ping-interval 60 --ws-ping-timeout 120
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
| `DATA_DIR` | `data` | Root for data artifacts |
| `SANDBOX_BASE` | `outputs/sandbox` | Where per-episode sandboxes are created |

## Health check
```bash
curl http://localhost:7860/health
# or
curl http://localhost:7860/info
```

## Prerequisites
- Download datasets with `python tools/download_datasets.py`
- `.venv` must be activated (or use `uvicorn` from the venv directly)
