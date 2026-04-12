#!/usr/bin/env bash
# Run full benchmark across multiple local GGUF models.
# Each model: start llama-server → wait for ready → run benchmark tasks → kill server.
#
# Usage:
#   ./run_benchmark.sh                    # all models, all categories, all difficulties
#   ./run_benchmark.sh --categories FP VR # subset of categories
#   ./run_benchmark.sh --difficulties easy medium
#   ./run_benchmark.sh --models "Qwen3.5-0.8B-UD-Q4_K_XL" "gemma-4-E2B-it-Q4_K_M"

set -euo pipefail

MODELS_DIR="${MODELS_DIR:-$HOME/models}"
PORT="${LLAMA_PORT:-8080}"
CONTEXT="${LLAMA_CONTEXT:-32768}"
BENCHMARK_CONFIG="tools/benchmark_config.yaml"
OUTPUT_DIR="outputs/benchmark"

# ── Model registry ─────────────────────────────────────────────────────────────
# Parallel arrays: name[i] → gguf[i]. Comment out to skip a model.
NAMES=(
  # "Qwen3.5-0.8B-UD-Q4_K_XL"
  # "Qwen3.5-2B-UD-Q4_K_XL"
  "gemma-4-E2B-it-Q4_K_M"
  "gemma-4-E4B-it-Q4_K_M"
  "Qwen3-4B-Q4_K_M"
  "Qwen3.5-9B-UD-Q4_K_XL"
)
GGUFS=(
  # "Qwen3.5-0.8B-UD-Q4_K_XL.gguf"
  # "Qwen3.5-2B-UD-Q4_K_XL.gguf"
  "gemma-4-E2B-it-Q4_K_M.gguf"
  "gemma-4-E4B-it-Q4_K_M.gguf"
  "Qwen3-4B-Q4_K_M.gguf"
  "Qwen3.5-9B-UD-Q4_K_XL.gguf"
)

# ── Parse CLI flags ────────────────────────────────────────────────────────────
EXTRA_ARGS=()
FILTER_MODELS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --models)
      shift
      while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
        FILTER_MODELS+=("$1")
        shift
      done
      ;;
    --categories|--difficulties|--datasets)
      flag="$1"; shift
      EXTRA_ARGS+=("$flag")
      while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
        EXTRA_ARGS+=("$1")
        shift
      done
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

# ── Server lifecycle helpers ───────────────────────────────────────────────────

wait_for_server() {
  local max_wait=120 elapsed=0
  while ! curl -s "http://localhost:${PORT}/health" >/dev/null 2>&1; do
    sleep 1; elapsed=$((elapsed + 1))
    if [ "$elapsed" -ge "$max_wait" ]; then
      echo "ERROR: llama-server failed to start within ${max_wait}s" >&2; return 1
    fi
  done
  while true; do
    status=$(curl -s "http://localhost:${PORT}/health" | grep -o '"status":"[^"]*"' | head -1)
    if [[ "$status" == *'"ok"'* ]]; then break; fi
    sleep 2; elapsed=$((elapsed + 2))
    if [ "$elapsed" -ge "$max_wait" ]; then
      echo "ERROR: model still loading after ${max_wait}s" >&2; return 1
    fi
    echo "  Still loading model (${elapsed}s)..."
  done
  echo "  Server ready (${elapsed}s)"
}

kill_server() {
  local pid="$1"
  if kill -0 "$pid" 2>/dev/null; then
    kill "$pid"; wait "$pid" 2>/dev/null || true
    echo "  Server stopped (pid $pid)"
  fi
}

# ── Pre-flight: check env server is running ────────────────────────────────────

ENV_PORT="${ENV_PORT:-7860}"
if ! curl -s "http://localhost:${ENV_PORT}/" >/dev/null 2>&1; then
  echo "ERROR: Environment server not running on port ${ENV_PORT}." >&2
  echo "  Start it first:" >&2
  echo "    uvicorn server.app:app --port ${ENV_PORT} --ws-ping-interval 60 --ws-ping-timeout 120" >&2
  exit 1
fi
echo "Environment server: OK (port ${ENV_PORT})"

# ── Pre-flight: check all model files exist ────────────────────────────────────

missing=0
for i in "${!NAMES[@]}"; do
  name="${NAMES[$i]}"
  gguf="${GGUFS[$i]}"

  # Skip if --models filter active and not in list
  if [ ${#FILTER_MODELS[@]} -gt 0 ]; then
    found=0
    for fm in "${FILTER_MODELS[@]}"; do
      if [[ "$name" == "$fm" ]]; then found=1; break; fi
    done
    if [ "$found" -eq 0 ]; then continue; fi
  fi

  if [ ! -f "${MODELS_DIR}/${gguf}" ]; then
    echo "MISSING: ${MODELS_DIR}/${gguf}" >&2
    missing=$((missing + 1))
  fi
done
if [ "$missing" -gt 0 ]; then
  echo "" >&2
  echo "ERROR: ${missing} model file(s) missing. Download them first:" >&2
  echo "  See docs/llama-cpp-setup.md for download commands." >&2
  exit 1
fi
echo "Model files: OK"
echo ""

# ── Kill any existing llama-server ─────────────────────────────────────────────

existing_pids=$(pgrep -f "llama-server" 2>/dev/null || true)
if [ -n "$existing_pids" ]; then
  echo "WARNING: Found existing llama-server process(es): $existing_pids"
  echo "  Killing them before starting benchmark..."
  kill $existing_pids 2>/dev/null || true
  sleep 2
  # Force-kill if still alive
  remaining=$(pgrep -f "llama-server" 2>/dev/null || true)
  if [ -n "$remaining" ]; then
    kill -9 $remaining 2>/dev/null || true
    sleep 1
  fi
  echo "  Cleared."
fi

# Also check if something else is already listening on our port
if curl -s "http://localhost:${PORT}/health" >/dev/null 2>&1; then
  echo "WARNING: Port ${PORT} is already in use by another service."
  echo "  Attempting to identify and kill it..."
  port_pid=$(lsof -ti :${PORT} 2>/dev/null || true)
  if [ -n "$port_pid" ]; then
    kill $port_pid 2>/dev/null || true
    sleep 2
    echo "  Killed PID $port_pid on port ${PORT}."
  else
    echo "  ERROR: Could not identify process on port ${PORT}. Exiting." >&2
    exit 1
  fi
fi

# ── Main loop ──────────────────────────────────────────────────────────────────

echo "============================================"
echo "Benchmark run: $(date)"
echo "Output: ${OUTPUT_DIR}"
echo "============================================"
echo ""

for i in "${!NAMES[@]}"; do
  name="${NAMES[$i]}"
  gguf="${GGUFS[$i]}"
  model_path="${MODELS_DIR}/${gguf}"

  # Skip if --models filter is active and this model isn't in the list
  if [ ${#FILTER_MODELS[@]} -gt 0 ]; then
    found=0
    for fm in "${FILTER_MODELS[@]}"; do
      if [[ "$name" == "$fm" ]]; then found=1; break; fi
    done
    if [ "$found" -eq 0 ]; then continue; fi
  fi

  if [ ! -f "$model_path" ]; then
    echo "SKIP: ${name} — file not found: ${model_path}"
    echo ""; continue
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
    echo "  FAILED — skipping ${name}"; echo ""; continue
  fi

  # Run benchmark for this model
  .venv/bin/python -m tools.benchmark_runner \
    --config "$BENCHMARK_CONFIG" \
    --model-name "$name" \
    --api-base "http://localhost:${PORT}/v1" \
    --api-key-env "OPENAI_API_KEY" \
    "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"

  # Shut down
  kill_server "$SERVER_PID"
  trap - EXIT
  echo ""
done

echo "============================================"
echo "Benchmark complete. Results in ${OUTPUT_DIR}/"
echo "============================================"
