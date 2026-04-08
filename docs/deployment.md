# Deployment Guide

## Local Development

### Prerequisites
- Python 3.11+
- pip

### Setup

```bash
git clone https://github.com/shivanshu1641/outlier_clean_data.git
cd outlier_clean_data
python -m venv .venv
source .venv/bin/activate
pip install -r server/requirements.txt
```

### Run Server

```bash
uvicorn server.app:app --port 8000 --ws-ping-interval 60 --ws-ping-timeout 120
```

### Run Inference

```bash
# Configure .env (see .env.example), then:
source .venv/bin/activate
python inference.py                    # all 6 tasks
python inference.py titanic_easy       # specific task
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `API_BASE_URL` | LLM API endpoint | `https://router.huggingface.co/v1` |
| `MODEL_NAME` | LLM model name | `Qwen/Qwen2.5-72B-Instruct` |
| `API_KEY` / `HF_TOKEN` | API token env var | (required) |
| `ENV_URL` | Environment server URL | `http://localhost:8000` |
| `MIN_CALL_INTERVAL` | Seconds between LLM calls | `2.5` |

## Docker (Local)

```bash
docker build -t data-cleaning-env .
docker run -p 8000:7860 data-cleaning-env
```

Note: Dockerfile uses port 7860 (HF Spaces default). Map to any local port.

## HF Spaces

**Space**: https://huggingface.co/spaces/shivshri/openenv_dataclean
**App**: https://shivshri-openenv-dataclean.hf.space

### Push updates

```bash
source .venv/bin/activate

# Validate first
openenv validate

# Option 1: openenv CLI (recommended — handles frontmatter + Dockerfile tweaks)
openenv push --repo-id shivshri/openenv_dataclean

# Option 2: git push directly
git push hf main --force
```

### Git remote setup (one-time)

```bash
git remote add hf https://shivshri:<HF_TOKEN>@huggingface.co/spaces/shivshri/openenv_dataclean
```

Get your token at https://huggingface.co/settings/tokens

### Test deployed endpoints

```bash
# Health
curl https://shivshri-openenv-dataclean.hf.space/

# Reset
curl -X POST https://shivshri-openenv-dataclean.hf.space/reset \
  -H "Content-Type: application/json" -d '{}'
```

### Important notes

- HF Spaces expects port **7860** (set in Dockerfile and server/app.py)
- `tasks/` and `data/` must be tracked in git (not gitignored)
- Binary files need Git LFS or should be removed
- `openenv push` is preferred over raw git push — it auto-adjusts README frontmatter
- Build logs: https://huggingface.co/spaces/shivshri/openenv_dataclean/logs/build

## Pre-Submission Checklist

- [x] `openenv validate` passes
- [x] `docker build .` succeeds
- [x] HF Space returns 200 on `GET /`
- [x] `POST /reset` with `{}` returns valid observation
- [x] `inference.py` at repo root with `[START]/[STEP]/[END]` stdout logs
- [x] Uses `from openai import OpenAI` for LLM calls
- [x] Uses `OpenAI(base_url=..., api_key=...)` with env-configured LLM settings
- [x] All 6 tasks produce rewards in [0.0, 1.0]
- [x] Runtime < 20 min on 2 vCPU / 8GB

## Troubleshooting

**Server won't start**: Check that `data/` and `tasks/` exist. Run `python tools/corruption/engine.py` to regenerate.

**Sandbox errors**: Make sure you're running from the venv. Check `outputs/sandbox/{episode}/scripts/` for saved scripts.

**HF Space stuck on "Building"**: Check build logs. Common issues: binary files rejected (use LFS), missing files in git.

**WebSocket timeout**: Increase ping interval/timeout in the uvicorn command or Dockerfile.
