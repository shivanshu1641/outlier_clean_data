# Runbook: Run Inference

## Command
```bash
source .venv/bin/activate
python inference.py                              # default eval set
EVAL_TASK_IDS=15 python inference.py             # 15 eval tasks
python inference.py titanic_easy                 # specific legacy task id
python inference.py titanic_easy wine_medium     # multiple legacy task ids
```

## Deployment-safe config

- Deployed and validator runs should use the injected `API_BASE_URL` and `API_KEY` variables when present.
- The script defaults `API_BASE_URL` to `https://router.huggingface.co/v1` and `MODEL_NAME` to `Qwen/Qwen2.5-72B-Instruct` if unset.
- `API_KEY` has no default; the script reads `API_KEY` first and falls back to `HF_TOKEN`.

## Configure your LLM provider

Edit `.env` with your provider settings:

### Local (Ollama)
```bash
API_BASE_URL=http://localhost:11434/v1
API_KEY=
MODEL_NAME=qwen3
```

### NVIDIA NIM
```bash
API_BASE_URL=https://integrate.api.nvidia.com/v1
API_KEY=nvapi-...
MODEL_NAME=nvidia/nemotron-super-49b-v1
```

### OpenAI
```bash
API_BASE_URL=https://api.openai.com/v1
API_KEY=sk-...
MODEL_NAME=gpt-4o
```

### Any other OpenAI-compatible endpoint
```bash
API_BASE_URL=http://your-vllm-server/v1
API_KEY=your-key-or-empty
MODEL_NAME=your-model-name
```

### Validator / hackathon environment
```bash
python inference.py
```

The validator injects `API_BASE_URL` and `API_KEY` for you. The key requirement is that `inference.py` passes env-configured values into `OpenAI(base_url=..., api_key=...)`. If they are unset outside validator runs, the script falls back only for `API_BASE_URL` and `MODEL_NAME`.

## Log levels

| `LOG_LEVEL` | What you see on stderr |
|-------------|----------------------|
| `INFO` | Per-step action + content preview, reward, errors fixed, LLM latency, token usage |
| `DEBUG` | Everything above + full LLM prompt/response, full observations |

```bash
LOG_LEVEL=DEBUG python inference.py titanic_easy
```

## Output
- **stdout**: Machine-readable JSON lines (`[START]`, `[STEP]`, `[END]`)
- **stderr**: Human-readable logs
- Supported environment actions include edit actions plus `undo` and `validate`

To capture only the structured output:
```bash
python inference.py 2>/dev/null
```

## Prerequisites
- Server must be running (`python server/app.py` or `uvicorn server.app:app --port 7860 --ws-ping-interval 60 --ws-ping-timeout 120`)
- Datasets should be downloaded from `catalog.json` with `python tools/download_datasets.py`
