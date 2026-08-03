#!/usr/bin/env bash
set -euo pipefail

# Use the Python environment that has torch/transformers/trl/deepspeed installed.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

export PYTHONPATH="${PROJECT_DIR}/Emu3.5:${PYTHONPATH:-}"

# NCCL timeout and diagnostics. Increase the timeout for large jobs and
# keep the async-error flags enabled so failures surface cleanly.
export NCCL_TIMEOUT=3600
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=0
export TORCH_NCCL_TRACE_BUFFER_SIZE=20000
# export NCCL_DEBUG=INFO    # Enable if you need to inspect collective failures
# export NCCL_P2P_DISABLE=1  # Enable only if your system has NVLink/P2P issues

NUM_GPUS="${NUM_GPUS:-8}"
MASTER_PORT="${MASTER_PORT:-29500}"
MODEL_PATH="${MODEL_PATH:-${PROJECT_DIR}/checkpoints/Emu3.5}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${PROJECT_DIR}/Emu3.5/src/tokenizer_emu3_ibq}"
TRAIN_DATA="${TRAIN_DATA:-${PROJECT_DIR}/Data/SFT/converted_v2/sft.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/outputs/sft}"
ATTN_IMPL="${ATTN_IMPL:-flash_attention_2}"

"${PYTHON_BIN}" -m torch.distributed.run \
  --nnodes=1 \
  --nproc_per_node="${NUM_GPUS}" \
  --master_addr=127.0.0.1 \
  --master_port="${MASTER_PORT}" \
  sft_agent.py \
  --model-path "${MODEL_PATH}" \
  --tokenizer-path "${TOKENIZER_PATH}" \
  --train-data "${TRAIN_DATA}" \
  --output-dir "${OUTPUT_DIR}" \
  --overwrite-output-dir \
  --deepspeed ds_config_zero2_bf16.json \
  --attn-implementation "${ATTN_IMPL}" \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --learning-rate 1e-5 \
  --num-train-epochs 1 \
  --max-seq-length 32768 \
  --dataloader-num-workers 1 \
  --bf16 \
  --tf32 \
  --optim adamw_torch
