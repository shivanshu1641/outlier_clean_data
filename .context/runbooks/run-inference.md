# Runbook: Run Inference

## Command
```bash
source .venv/bin/activate
python inference.py                                  # all 18 eval tasks
python inference.py titanic easy                     # single task (default fmt=csv)
python inference.py titanic easy json                # explicit format
python inference.py titanic/easy/json                # slash syntax also works
```

## Eval suite
`inference.py` defines an 18-task eval suite (`EVAL_TASKS`) as `(dataset_id, difficulty, format)` triples — 3 per dataset across 6 datasets: titanic, iris, boston_housing, diabetes, wine_quality, breast_cancer.

| Dataset | Easy | Medium | Hard |
|---------|------|--------|------|
| Titanic | csv | csv | csv |
| Iris | csv | csv, jsonl | — |
| Boston Housing | — | csv | csv, json |
| Diabetes | — | csv | csv, json |
| Wine Quality | csv | csv | csv |
| Breast Cancer | csv | csv, jsonl | — |

## Benchmark runner
For systematic model evaluation across categories:
```bash
python tools/benchmark_runner.py                              # default config
python tools/benchmark_runner.py --models gpt-4o --categories FP VR
python tools/benchmark_runner.py --config tools/benchmark_config.yaml
```
Results saved to `outputs/benchmark/` as JSONL + CSV.

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

Tested local models (via Ollama):

| Model | Ollama tag | Notes |
|-------|-----------|-------|
| Qwen 3 4B | `qwen3` | Fast, good for easy/medium tasks |
| Qwen 3 8B | `qwen3:8b` | Better on hard tasks, ~2× slower |
| Gemma 3 4B | `gemma3:4b` | Comparable to Qwen 3 4B on data cleaning |
| Gemma 3 12B | `gemma3:12b` | Stronger reasoning, higher RAM usage |
| Llama 3.2 3B | `llama3.2:3b` | Smaller, struggles on hard tasks |
| Llama 3.1 8B | `llama3.1:8b` | Solid general-purpose, good on SR/SV |
| Phi-4 Mini | `phi4-mini` | Compact, decent on structured data |
| Phi-4 | `phi4` | Better than mini, slower |
| Mistral 7B | `mistral` | Good baseline for benchmarking |
| DeepSeek-R1 7B | `deepseek-r1:7b` | Chain-of-thought, good on complex tasks |

Pull a model: `ollama pull qwen3`

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

### OpenRouter (free tier)
```bash
API_BASE_URL=https://openrouter.ai/api/v1
API_KEY=sk-or-...
MODEL_NAME=google/gemma-3-27b-it:free
```

### HuggingFace Inference
```bash
API_BASE_URL=https://router.huggingface.co/v1
API_KEY=hf_...
MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
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
