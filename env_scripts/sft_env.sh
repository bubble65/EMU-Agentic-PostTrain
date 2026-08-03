#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

pip install torch==2.8.0
pip install --no-cache-dir https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
cd "${PROJECT_DIR}/Emu3.5/requirements"
pip install -r vllm.txt
pip install transformers==4.55.2
pip install datasets
pip install nvitop
pip install trl==0.9.6
pip install deepspeed
pip install pillow
pip install tiktoken
pip install omegaconf
pip install rich
pip install fastapi
pip install uvicorn
