#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

export ARK_API_KEY="ark-xxx"
export ARK_MODEL="doubao-seed-2-0-pro-260215"
export SERPER_API_KEY="xxx"
export GEMINI_API_KEY="${GEMINI_API_KEY:-}"
export GEMINI_BASE_URL="${GEMINI_BASE_URL:-https://generativelanguage.googleapis.com/v1beta}"
export GEMINI_MODEL="${GEMINI_MODEL:-gemini-3-pro-image}"
export GEMINI_ASPECT_RATIO="${GEMINI_ASPECT_RATIO:-4:3}"
export GEMINI_IMAGE_SIZE="${GEMINI_IMAGE_SIZE:-1K}"
export GEMINI_RESPONSE_MIME_TYPE="${GEMINI_RESPONSE_MIME_TYPE:-image/png}"
export MODEL_NAME="doubao2.0"
export DATASET="gen_sft"
export OUTPUT_PATH="./outputs"
export MAX_ITEMS=1

rm -rf "${OUTPUT_PATH:?}/${MODEL_NAME}/${DATASET}"

python3 -u run_multi_react.py
