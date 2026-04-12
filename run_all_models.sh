#!/usr/bin/env bash
# Run inference across multiple local models sequentially.
# Each model: start llama-server → wait for ready → run inference → kill server.

set -euo pipefail

MODELS_DIR="$HOME/models"
PORT=8080
CONTEXT=32768
# Default tasks if none given on CLI — use exact catalog keys
if [ $# -gt 0 ]; then
  TASKS=("$@")
else
  TASKS=(titanic/easy/csv titanic/medium/csv wine_quality/easy/csv wine_quality/medium/csv titanic/hard/csv wine_quality/hard/csv)
fi

# Parallel arrays — name[i] maps to gguf[i]. Comment out to skip.
NAMES=(
  "Qwen3.5-0.8B-UD-Q4_K_XL"
  "Qwen3.5-2B-UD-Q4_K_XL"
  "gemma-4-E2B-it-Q4_K_M"
  "gemma-4-E4B-it-Q4_K_M"
  "Qwen3-4B-Q4_K_M"
  "Qwen3.5-9B-UD-Q4_K_XL"
)
GGUFS=(
  "Qwen3.5-0.8B-UD-Q4_K_XL.gguf"
  "Qwen3.5-2B-UD-Q4_K_XL.gguf"
  "gemma-4-E2B-it-Q4_K_M.gguf"
  "gemma-4-E4B-it-Q4_K_M.gguf"
  "Qwen3-4B-Q4_K_M.gguf"
  "Qwen3.5-9B-UD-Q4_K_XL.gguf"
)

wait_for_server() {
  local max_wait=120
  local elapsed=0
  # Phase 1: wait for HTTP to respond
  while ! curl -s "http://localhost:${PORT}/health" >/dev/null 2>&1; do
    sleep 1
    elapsed=$((elapsed + 1))
    if [ "$elapsed" -ge "$max_wait" ]; then
      echo "ERROR: llama-server failed to start within ${max_wait}s" >&2
      return 1
    fi
  done
  # Phase 2: wait for model to finish loading (health returns {"status":"ok"})
  while true; do
    status=$(curl -s "http://localhost:${PORT}/health" | grep -o '"status":"[^"]*"' | head -1)
    if [[ "$status" == *'"ok"'* ]]; then
      break
    fi
    sleep 2
    elapsed=$((elapsed + 2))
    if [ "$elapsed" -ge "$max_wait" ]; then
      echo "ERROR: model still loading after ${max_wait}s" >&2
      return 1
    fi
    echo "  Still loading model (${elapsed}s)..."
  done
  echo "  Server ready (${elapsed}s)"
}

kill_server() {
  local pid="$1"
  if kill -0 "$pid" 2>/dev/null; then
    kill "$pid"
    wait "$pid" 2>/dev/null || true
    echo "  Server stopped (pid $pid)"
  fi
}

echo "============================================"
echo "Multi-model benchmark: tasks=${TASKS[*]}"
echo "============================================"
echo ""

for i in "${!NAMES[@]}"; do
  name="${NAMES[$i]}"
  gguf="${GGUFS[$i]}"
  model_path="${MODELS_DIR}/${gguf}"

  if [ ! -f "$model_path" ]; then
    echo "SKIP: ${name} — file not found: ${model_path}"
    echo ""
    continue
  fi

  echo "──────────────────────────────────────────"
  echo "Model: ${name}"
  echo "──────────────────────────────────────────"

  # Start llama-server in background
  llama-server \
    -m "$model_path" \
    -c "$CONTEXT" \
    --port "$PORT" \
    -ngl 99 \
    --reasoning-budget 0 \
    >/dev/null 2>&1 &
  SERVER_PID=$!
  trap "kill_server $SERVER_PID" EXIT

  echo "  Starting server (pid ${SERVER_PID})..."
  if ! wait_for_server; then
    kill_server "$SERVER_PID"
    echo "  FAILED — skipping ${name}"
    echo ""
    continue
  fi

  # Run inference (call python directly — inference.sh hardcodes MODEL_NAME)
  MODEL_NAME="$name" \
  API_BASE_URL="http://localhost:${PORT}/v1" \
  OPENAI_API_KEY="not-needed" \
  ENV_URL="http://localhost:7860" \
  MIN_CALL_INTERVAL=0 \
    .venv/bin/python inference.py "${TASKS[@]}"

  # Shut down
  kill_server "$SERVER_PID"
  trap - EXIT
  echo ""
done

echo "============================================"
echo "All models complete."
echo "============================================"
