# Runbook: Setup llama.cpp on macOS

Local LLM inference for testing/development without API rate limits or costs.

Tested on: Apple Silicon (M1 Pro 16GB+). Works on Intel Macs too (slower).

## Install

```bash
# Install via Homebrew
brew install llama.cpp

# If already installed, upgrade to latest
brew upgrade llama.cpp

# Verify
llama-server --version
```

## Download a model

**Recommended: Gemma 4 E2B** — Google's latest, 2B params, runs great on 16GB M1 Pro.

```bash
# Install huggingface CLI if needed
pip install huggingface-hub

# Gemma 4 E2B Q4_K_M (recommended) — 3.1GB, best quality/speed tradeoff
hf download unsloth/gemma-4-E2B-it-GGUF gemma-4-E2B-it-Q4_K_M.gguf --local-dir ~/models/
```

### Other Gemma 4 E2B quantizations

| Quant        | Size       | Notes                                   |
| ------------ | ---------- | --------------------------------------- |
| `UD-IQ2_M`   | 2.3 GB     | Fastest, lowest quality                 |
| `Q3_K_M`     | 2.5 GB     | Good for constrained RAM                |
| **`Q4_K_M`** | **3.1 GB** | **Recommended — best tradeoff**         |
| `Q8_0`       | 5.0 GB     | Highest quality, still fast on M1 Pro   |
| `BF16`       | 9.3 GB     | Full precision — avoid on 16GB machines |

### Benchmark model suite

All models used by `run_benchmark.sh`. Download all of them to `~/models/`:

```bash
# Qwen3.5-0.8B (~1GB)
hf download unsloth/Qwen3.5-0.8B-GGUF \
  Qwen3.5-0.8B-UD-Q4_K_XL.gguf --local-dir ~/models/

# Qwen3.5-2B (~2GB)
hf download unsloth/Qwen3.5-2B-GGUF \
  Qwen3.5-2B-UD-Q4_K_XL.gguf --local-dir ~/models/

# gemma-4-E2B (~3GB)
hf download unsloth/gemma-4-E2B-it-GGUF \
  gemma-4-E2B-it-Q4_K_M.gguf --local-dir ~/models/

# gemma-4-E4B (~5GB)
hf download unsloth/gemma-4-E4B-it-GGUF \
  gemma-4-E4B-it-Q4_K_M.gguf --local-dir ~/models/

# Qwen3-4B (~3GB)
hf download Qwen/Qwen3-4B-GGUF \
  Qwen3-4B-Q4_K_M.gguf --local-dir ~/models/

# Qwen3.5-9B (~7GB)
hf download unsloth/Qwen3.5-9B-GGUF \
  Qwen3.5-9B-UD-Q4_K_XL.gguf --local-dir ~/models/
```

### Other models

```bash
# Qwen 2.5 3B — 2GB, strong reasoning
hf download Qwen/Qwen2.5-3B-Instruct-GGUF qwen2.5-3b-instruct-q4_k_m.gguf --local-dir ~/models/
```

## Run the server

```bash
# Apple Silicon (Metal GPU acceleration — recommended)
llama-server -m ~/models/gemma-4-E2B-it-Q4_K_M.gguf -c 8192 --port 8080 -ngl 99

# CPU only (Intel Macs or fallback)
llama-server -m ~/models/gemma-4-E2B-it-Q4_K_M.gguf -c 8192 --port 8080

# Flags:
#   -m      model path
#   -c      context window size (8192 is enough for this project)
#   --port  server port
#   -ngl    layers offloaded to GPU (99 = all — use this on Apple Silicon)
```

The server exposes an OpenAI-compatible API at `http://localhost:8080/v1`.

**Note on Metal GPU:** GGUF files are hardware-agnostic — there's no special "Metal-tuned" variant. Metal acceleration comes from llama.cpp itself via the `-ngl 99` flag, which offloads all model layers to the Apple Silicon GPU. Any GGUF file works with Metal.

## Configure inference

```
API_BASE_URL=http://localhost:8080/v1
OPENAI_API_KEY=dummy
MODEL_NAME=gemma-4-E2B-it
ENV_URL=http://localhost:7860
MIN_CALL_INTERVAL=0
```

`MIN_CALL_INTERVAL=0` since there are no rate limits locally.

## Verify it works

```bash
# Quick test
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "gemma-4-E2B-it", "messages": [{"role": "user", "content": "Say hello"}]}'

# Run one task
python inference.py titanic_easy
```

## RAM requirements

| Model           | Quant  | RAM needed | 16GB M1 Pro? |
| --------------- | ------ | ---------- | ------------ |
| Gemma 4 E2B     | Q4_K_M | ~4GB       | Yes          |
| Gemma 4 E2B     | Q8_0   | ~6GB       | Yes          |
| Gemma 4 E4B     | Q4_K_M | ~5GB       | Yes          |
| Gemma 4 26B-A4B | Q4_K_M | ~17GB      | No           |

Rule of thumb: model file size + ~1GB overhead. Stay under 10GB total to leave room for OS + server + inference script.

## Troubleshooting

| Issue                             | Fix                                                      |
| --------------------------------- | -------------------------------------------------------- |
| `command not found: llama-server` | `brew install llama.cpp`                                 |
| `command not found: hf`           | `pip install huggingface-hub`                            |
| `GatedRepoError` / 401            | Run `hf login`, then accept model license on HuggingFace |
| Slow inference on Intel Mac       | Use a smaller quant (`Q3_K_M`) or reduce `-c` to 2048    |
| Out of memory                     | Use smaller quant or smaller model                       |
| Model not found error             | Check `-m` path; file should end in `.gguf`              |

## Alternative: Ollama

If you prefer a simpler setup (no HF auth, auto-manages models):

```bash
# Install
brew install ollama

# Pull and run Gemma 3 (Gemma 4 may be available via ollama pull gemma4:2b)
ollama pull gemma3:1b
ollama serve  # runs on port 11434

# inference env
API_BASE_URL=http://localhost:11434/v1
OPENAI_API_KEY=dummy
MODEL_NAME=gemma3:1b
ENV_URL=http://localhost:7860
MIN_CALL_INTERVAL=0
```
