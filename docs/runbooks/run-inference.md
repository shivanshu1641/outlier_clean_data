# Runbook: Run Inference

## Command
```bash
source .venv/bin/activate
python inference.py                          # all 6 tasks
python inference.py titanic_easy             # specific task
python inference.py titanic_easy wine_medium # multiple tasks
```

## Configure your LLM provider

Edit `.env` with your provider settings:

### Local (Ollama)
```
API_BASE_URL=http://localhost:11434/v1
API_KEY=
MODEL_NAME=qwen3
```

### NVIDIA NIM
```
API_BASE_URL=https://integrate.api.nvidia.com/v1
API_KEY=nvapi-...
MODEL_NAME=nvidia/nemotron-super-49b-v1
```

### OpenAI
```
API_BASE_URL=https://api.openai.com/v1
API_KEY=sk-...
MODEL_NAME=gpt-4o
```

### Any other OpenAI-compatible endpoint
```
API_BASE_URL=http://your-vllm-server/v1
API_KEY=your-key-or-empty
MODEL_NAME=your-model-name
```

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

To capture only the structured output:
```bash
python inference.py 2>/dev/null
```

## Prerequisites
- Server must be running (`uvicorn server.app:app --port 8000 --ws-ping-interval 60 --ws-ping-timeout 120`)
- Task artifacts must exist (`python tools/corruption/engine.py`)
