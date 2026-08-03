#!/usr/bin/env bash
# Start the Emu3.5 SFT inference server (server.py).
#
# Override anything below via env. Defaults are repo-relative public paths;
# adjust EMU_MODEL_PATH / EMU_VQ_PATH as needed.
#
# Required:
#   EMU_MODEL_PATH       SFT'd Emu3.5 dir
#   EMU_VQ_PATH          vision tokenizer dir
#
# Optional:
#   EMU_TOKENIZER_PATH   text tokenizer (default Emu3.5/src/tokenizer_emu3_ibq)
#   EMU_ROOT             root of the upstream Emu3.5 repo (so src.utils.* imports work)
#   EMU_TP_SIZE          tensor-parallel size (default 2)
#   EMU_GPU_MEM_UTIL     gpu memory util      (default 0.7)
#   EMU_VQ_DEVICE        device for vq        (default cuda:0)
#   EMU_HOST/EMU_PORT    bind addr            (default 0.0.0.0:23333)
#   EMU_IMAGE_SAVE_DIR   where generated PNGs go
#   EMU_MAX_NEW_TOKENS   default max output tokens (default 8192)
#   EMU_TEXT_CFG         CFG for text passes  (default 1.0 = no CFG)
#   EMU_IMAGE_CFG        CFG for image pass   (default 3.0; emu3.5 interleaved recipe)
#   EMU_IMAGE_AREA       target image area    (default 1048576 ≈ 1024x1024)
#   EMU_MAX_IMAGE_TOKENS image-pass max tokens (default 8192)
#   EMU_SEED             vllm seed            (default 6666)
#   CUDA_VISIBLE_DEVICES

# B300 (sm_103a) needs CUDA 13+ ptxas. If a CUDA 13 install exists, point
# TRITON_PTXAS_PATH at it BEFORE running. Otherwise we skip torch.compile
# entirely below (VLLM_USE_V1=1 + enforce_eager via env) so ptxas is never invoked.
if [[ -n "${TRITON_PTXAS_PATH:-}" && ! -x "${TRITON_PTXAS_PATH}" ]]; then
    echo "[start_server] WARN: TRITON_PTXAS_PATH=${TRITON_PTXAS_PATH} not executable, unsetting"
    unset TRITON_PTXAS_PATH
fi

# Disable torch.compile / inductor so the bundled ptxas (CUDA 12.8, no sm_103a) isn't called.
export VLLM_DISABLE_COMPILE_CACHE="${VLLM_DISABLE_COMPILE_CACHE:-1}"
export EMU_ENFORCE_EAGER="${EMU_ENFORCE_EAGER:-1}"

# enforce_eager only kills vLLM's piecewise compile; some vllm modules (e.g.
# VocabParallelEmbedding.get_masked_input_and_mask) are wrapped with bare
# @torch.compile and still hit Inductor -> Triton -> ptxas. Kill dynamo
# globally to keep ptxas out of the loop on B300 / sm_103a.
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"
export TORCHINDUCTOR_FORCE_DISABLE_CACHES="${TORCHINDUCTOR_FORCE_DISABLE_CACHES:-1}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

# ── Proxy ────────────────────────────────────────────────────────────────
# Optional outbound proxy for image download/search traffic.
if [[ -n "${PROXY_URL:-}" ]]; then
    export http_proxy="${http_proxy:-$PROXY_URL}"
    export https_proxy="${https_proxy:-$PROXY_URL}"
    export ftp_proxy="${ftp_proxy:-$PROXY_URL}"
    export HTTP_PROXY="${HTTP_PROXY:-$PROXY_URL}"
    export HTTPS_PROXY="${HTTPS_PROXY:-$PROXY_URL}"
    export FTP_PROXY="${FTP_PROXY:-$PROXY_URL}"
fi
export no_proxy="${no_proxy:-localhost,127.0.0.1}"
export NO_PROXY="${NO_PROXY:-$no_proxy}"

set -euo pipefail

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$THIS_DIR/.." && pwd)"

: "${EMU_ROOT:=$PROJECT_ROOT/Emu3.5}"
: "${EMU_MODEL_PATH:=$PROJECT_ROOT/checkpoints/emu3p5-sft}"
: "${EMU_VQ_PATH:=$PROJECT_ROOT/checkpoints/Emu3.5-VisionTokenizer}"
: "${EMU_TOKENIZER_PATH:=$EMU_ROOT/src/tokenizer_emu3_ibq}"
: "${EMU_TP_SIZE:=2}"
: "${EMU_GPU_MEM_UTIL:=0.7}"
: "${EMU_VQ_DEVICE:=cuda:0}"
: "${EMU_HOST:=0.0.0.0}"
: "${EMU_PORT:=23333}"
: "${EMU_IMAGE_SAVE_DIR:=$THIS_DIR/workspace/images}"
: "${EMU_MAX_NEW_TOKENS:=8192}"
: "${EMU_TEXT_CFG:=1.0}"
: "${EMU_IMAGE_CFG:=3.0}"
: "${EMU_IMAGE_AREA:=1048576}"
: "${EMU_MAX_IMAGE_TOKENS:=8192}"
: "${EMU_SEED:=6666}"

export EMU_ROOT EMU_MODEL_PATH EMU_VQ_PATH EMU_TOKENIZER_PATH
export EMU_TP_SIZE EMU_GPU_MEM_UTIL EMU_VQ_DEVICE EMU_HOST EMU_PORT
export EMU_IMAGE_SAVE_DIR EMU_MAX_NEW_TOKENS EMU_TEXT_CFG EMU_IMAGE_CFG
export EMU_IMAGE_AREA EMU_MAX_IMAGE_TOKENS EMU_SEED

export PYTHONPATH="${PYTHONPATH:-}:$PROJECT_ROOT:$EMU_ROOT:$THIS_DIR"

echo "[start_server] PYTHONPATH=$PYTHONPATH"
echo "[start_server] model=$EMU_MODEL_PATH"
echo "[start_server] tokenizer=$EMU_TOKENIZER_PATH"
echo "[start_server] vq=$EMU_VQ_PATH (device=$EMU_VQ_DEVICE)"
echo "[start_server] listen=$EMU_HOST:$EMU_PORT, tp=$EMU_TP_SIZE, gpu_mem=$EMU_GPU_MEM_UTIL"
echo "[start_server] image_save_dir=$EMU_IMAGE_SAVE_DIR"
echo "[start_server] text_cfg=$EMU_TEXT_CFG image_cfg=$EMU_IMAGE_CFG image_area=$EMU_IMAGE_AREA"

cd "$EMU_ROOT"   # so `src.utils.*` resolves cleanly
python -u "$THIS_DIR/server.py"
