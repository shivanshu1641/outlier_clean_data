# Runbook: Deployment

## HuggingFace Spaces (primary)

The app is deployed as a Gradio Space on HuggingFace at port 7860.

### Build & push

```bash
# From project root
docker build -t outlier-clean-data .
docker tag outlier-clean-data registry.hf.space/<your-space>:latest
docker push registry.hf.space/<your-space>:latest
```

Or push directly via git if the Space is linked to this repo:
```bash
git push hf main
```

### What gets bundled in the image

- `server/` — FastAPI WebSocket environment
- `ui/` — Gradio tabs (Leaderboard, Episode Explorer, Catalog)
- `tools/` — benchmark runner, download script, catalog enricher
- `datasets/catalog.json` — 25-dataset catalog (no CSVs; downloaded on first use)
- `outputs/` — pre-computed benchmark results (optional)

**Note:** `data/clean/` CSVs are NOT bundled. The server downloads them on first `reset()` call via `tools/download_datasets.py` fallback logic (GitHub mirrors → source_url).

### Environment variables (set in Space secrets)

| Var | Required | Description |
|-----|----------|-------------|
| `API_BASE_URL` | No | LLM endpoint (default: HF router) |
| `API_KEY` / `HF_TOKEN` | Yes | Auth for LLM calls |
| `MODEL_NAME` | No | Default: `Qwen/Qwen2.5-72B-Instruct` |
| `DATA_DIR` | No | Default: `data` |
| `SANDBOX_BASE` | No | Default: `outputs/sandbox` |

### Health check

```bash
curl https://<your-space>.hf.space/health
curl https://<your-space>.hf.space/info
```

## Local Docker

```bash
docker build -t outlier-clean-data .
docker run -p 7860:7860 \
  -e API_KEY=hf_... \
  -e MODEL_NAME=Qwen/Qwen2.5-72B-Instruct \
  outlier-clean-data
```

## Local dev (no Docker)

```bash
source .venv/bin/activate
python tools/download_datasets.py          # pre-fetch clean CSVs
uvicorn server.app:app --port 7860 \
  --ws-ping-interval 60 --ws-ping-timeout 120
```

Open `http://localhost:7860` for the Gradio UI.
