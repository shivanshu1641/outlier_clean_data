#!/usr/bin/env bash
# Run inference against a local llama-server.
# Usage: ./inference.sh [task args...]
#   ./inference.sh                        # all tasks
#   ./inference.sh titanic/easy/csv       # single task

set -euo pipefail

export API_BASE_URL="${API_BASE_URL:-http://localhost:8080/v1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"
export MODEL_NAME="${MODEL_NAME:-gemma-4-E2B-it-Q4_K_M}"
export ENV_URL="${ENV_URL:-http://localhost:7860}"
export MIN_CALL_INTERVAL="${MIN_CALL_INTERVAL:-0}"

# ── Pre-flight checks ─────────────────────────────────────────────────────────

# Check env server
if ! curl -s "${ENV_URL}/" >/dev/null 2>&1; then
  echo "ERROR: Environment server not running at ${ENV_URL}" >&2
  echo "  Start it first:" >&2
  echo "    uvicorn server.app:app --port 7860 --ws-ping-interval 60 --ws-ping-timeout 120" >&2
  exit 1
fi
echo "Environment server: OK (${ENV_URL})"

# Check LLM API
if ! curl -s "${API_BASE_URL%/v1}/health" >/dev/null 2>&1; then
  echo "ERROR: LLM server not running at ${API_BASE_URL}" >&2
  echo "  Start llama-server first:" >&2
  echo "    llama-server -m ~/models/<model>.gguf -c 32768 --port 8080 -ngl 99" >&2
  echo "  See docs/setup-llama-mac.md for model downloads." >&2
  exit 1
fi
echo "LLM server: OK (${API_BASE_URL})"
echo ""

# ── Run ────────────────────────────────────────────────────────────────────────

.venv/bin/python inference.py "$@"
