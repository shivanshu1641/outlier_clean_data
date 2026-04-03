# Deployment Guide

## Local Development

### Prerequisites
- Python 3.11+
- pip or uv

### Setup

```bash
# Clone and enter repo
git clone https://github.com/shivanshu1641/outlier_clean_data.git
cd outlier_clean_data

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r server/requirements.txt

# (Optional) Regenerate dirty data
python tools/corruption/engine.py
```

### Run Server

```bash
uvicorn server.app:app --port 8000
```

Verify: `curl http://localhost:8000/health` → `{"status": "healthy"}`

### Run Inference

```bash
# With a local LLM (e.g., Ollama)
API_BASE_URL=http://localhost:11434/v1 MODEL_NAME=qwen3 ENV_URL=http://localhost:8000 python inference.py

# With OpenAI
API_BASE_URL=https://api.openai.com/v1 MODEL_NAME=gpt-4 HF_TOKEN=sk-xxx ENV_URL=http://localhost:8000 python inference.py

# Run a single task
python inference.py easy_titanic
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `API_BASE_URL` | LLM API endpoint | `http://localhost:11434/v1` |
| `MODEL_NAME` | LLM model name | `qwen3` |
| `HF_TOKEN` | Hugging Face / API token | (empty) |
| `ENV_URL` | Environment server URL | `http://localhost:8000` |
| `TASKS_DIR` | Tasks config directory | `tasks` |
| `DATA_DIR` | Data directory | `data` |
| `SANDBOX_BASE` | Sandbox output directory | `outputs/sandbox` |

## Docker

### Build

```bash
docker build -f server/Dockerfile -t data-cleaning-env .
```

### Run

```bash
docker run -p 8000:8000 data-cleaning-env
```

### Verify

```bash
curl http://localhost:8000/health
curl http://localhost:8000/schema | python -m json.tool
```

## Hugging Face Spaces

### Option 1: openenv push

```bash
pip install openenv-core
openenv push --repo-id YOUR_USERNAME/data-cleaning-env
```

### Option 2: Manual

1. Create a new Space on huggingface.co (Docker type)
2. Upload all files (models.py, server/, tasks/, data/, openenv.yaml, etc.)
3. Ensure the Dockerfile is at `server/Dockerfile`
4. Space should auto-build and deploy

### Verify Deployment

```bash
# Health check
curl https://YOUR_USERNAME-data-cleaning-env.hf.space/health

# Reset test
curl -X POST https://YOUR_USERNAME-data-cleaning-env.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "easy_titanic"}'
```

## Pre-Submission Validation Checklist

- [ ] `docker build -f server/Dockerfile -t test .` succeeds
- [ ] `docker run -p 8000:8000 test` starts without errors
- [ ] `curl http://localhost:8000/health` returns 200
- [ ] `/reset` with each task_id returns valid observation
- [ ] `/ws` WebSocket connection works
- [ ] `openenv.yaml` has spec_version: 1
- [ ] `inference.py` runs end-to-end and produces [START]/[STEP]/[END] logs
- [ ] All 3 tasks produce rewards in [0.0, 1.0]
- [ ] Runtime < 20 minutes on 2 vCPU / 8GB

## Troubleshooting

**Server won't start**: Check that `data/dirty/*.csv` and `tasks/*.json` exist. Run `python tools/corruption/engine.py` if missing.

**Sandbox errors**: The subprocess uses `sys.executable` — make sure you're running from the venv. Check `outputs/sandbox/{episode}/scripts/` for saved scripts.

**Memory issues**: Large datasets + many episodes can fill `outputs/sandbox/`. Clean up old episodes periodically.

**WebSocket timeout**: Increase `message_timeout_s` in the client constructor if transforms take long.
