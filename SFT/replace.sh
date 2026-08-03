#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DATA_DIR="${DATA_DIR:-${PROJECT_DIR}/Data/SFT}"

INPUT="${INPUT:-${DATA_DIR}/raw_rollout.jsonl}"
OUTPUT="${OUTPUT:-${DATA_DIR}/normalized_rollout.jsonl}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/replace_pe.py" \
  --input "${INPUT}" \
  --output "${OUTPUT}"
