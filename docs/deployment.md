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
ENABLE_WEB_INTERFACE=true uvicorn server.app:app --port 7860 --ws-ping-interval 60 --ws-ping-timeout 120
```

### Run Inference (local model via llama-server)

```bash
# Start llama-server first (see docs/setup-llama-mac.md for model downloads)
llama-server -m ~/models/gemma-4-E2B-it-Q4_K_M.gguf -c 8192 --port 8080 -ngl 99

# Via wrapper script (pre-flight checks for env server + LLM API)
./inference.sh                              # all 18 eval tasks
./inference.sh titanic/easy/csv             # single task

# Or directly
python inference.py
python inference.py titanic_easy wine_medium
```

### Run Inference (paid API — OpenAI, Groq, etc.)

```bash
export OPENAI_API_KEY="sk-..."
API_BASE_URL="https://api.openai.com/v1" MODEL_NAME="gpt-4o" MIN_CALL_INTERVAL=2.5 python inference.py

# Groq
export OPENAI_API_KEY="gsk-..."
API_BASE_URL="https://api.groq.com/openai/v1" MODEL_NAME="llama-3.3-70b" python inference.py
```

### Run Benchmark (multi-model, local GGUFs)

Automatically starts/stops llama-server per model. Requires env server running + model GGUFs in `~/models/`.

```bash
# Full benchmark: 6 models × all datasets × 3 difficulties
./run_benchmark.sh

# Filter by model/category/difficulty
./run_benchmark.sh --models "gemma-4-E2B-it-Q4_K_M" --categories FP VR --difficulties easy medium

# Random sample of N tasks (skips already-completed)
./run_benchmark.sh --max-tasks 50

# Benchmark models: Qwen3.5-0.8B/2B/9B, gemma-4-E2B/E4B, Qwen3-4B
# See docs/setup-llama-mac.md for download commands
```

### Run Benchmark (paid API)

```bash
# --api-key-env is the NAME of the env var, not the key itself
export OPENAI_API_KEY="sk-..."
python -m tools.benchmark_runner --model-name "gpt-4o" --api-base "https://api.openai.com/v1" --api-key-env "OPENAI_API_KEY" --max-tasks 20

# Groq example
export OPENAI_API_KEY="gsk-..."
python -m tools.benchmark_runner --model-name "llama-3.3-70b" --api-base "https://api.groq.com/openai/v1" --api-key-env "OPENAI_API_KEY" --max-tasks 20
```

Benchmark config: `tools/benchmark_config.yaml` — set `max_tasks`, `min_call_interval`, models, categories, difficulties.

Results saved to `outputs/benchmark/{results.jsonl, summary.csv, episodes/*.jsonl}`. Resumable — skips already-completed (dataset, category, difficulty, model, seed) combos.

### Launch UI (Gradio Dashboard)

```bash
python -m ui.app                    # default port 7861
python -m ui.app --port 7862        # custom port
# Also available at /web when env server runs with ENABLE_WEB_INTERFACE=true
```

### Environment Variables

| Variable                      | Description                            | Default                     |
| ----------------------------- | -------------------------------------- | --------------------------- |
| `API_BASE_URL`                | LLM API endpoint                       | `https://api.openai.com/v1` |
| `MODEL_NAME`                  | LLM model name                         | `gpt-4o`                    |
| `OPENAI_API_KEY` / `HF_TOKEN` | API token env var (`HF_TOKEN` wins)    | empty                       |
| `ENV_URL`                     | Environment server URL                 | `http://localhost:7860`     |
| `MIN_CALL_INTERVAL`           | Seconds between LLM calls              | `2.5`                       |
| `LOG_LEVEL`                   | Inference log verbosity                | `INFO`                      |
| `LOG_DIR`                     | Directory for JSONL and CSV run output | `outputs/logs`              |

`inference.sh` is the local convenience wrapper used in this repo. It currently points at `gemma-4-E2B-it` on a local OpenAI-compatible endpoint and overrides several of the defaults above for local runs.

## Docker (Local)

```bash
docker build -t outlier-clean-data .
docker run -p 7860:7860 outlier-clean-data
```

Use the root `Dockerfile` for submission. It serves the environment on port 7860. Run `inference.py` outside the container.

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
- [x] All configured tasks produce rewards in [0.0, 1.0]
- [x] Runtime < 20 min on 2 vCPU / 8GB

## Troubleshooting

**Server won't start**: Check that `data/` exists. Run `python tools/download_datasets.py` if the clean datasets are missing.

**Sandbox errors**: Make sure you're running from the venv. Check `outputs/sandbox/{episode}/scripts/` for saved scripts.

**HF Space stuck on "Building"**: Check build logs. Common issues: binary files rejected (use LFS), missing files in git.

**WebSocket timeout**: Increase ping interval/timeout in the uvicorn command or Dockerfile.
