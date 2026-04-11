#!/usr/bin/env bash

export API_BASE_URL=http://localhost:16384/v1
export OPENAI_API_KEY=dummy
export MODEL_NAME=gemma-4-E2B-it
export ENV_URL=http://localhost:7860
export MIN_CALL_INTERVAL=0

.venv/bin/python inference.py "$@"
