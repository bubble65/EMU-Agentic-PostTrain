#!/usr/bin/env bash
# Drive emu_infer/run.py over a single jsonl dataset and dump results.
#
# Prereq: the emu server is up (start_server.sh in another shell on the GPU box).
#
# Usage:
#   DATASET_NAME=WISE bash emu_infer/run_agent.sh
#   bash emu_infer/run_agent.sh WISE
#   bash emu_infer/run_agent.sh /path/to/dataset.jsonl /path/to/output
#
# Defaults:
#   DATA_ROOT    = ../Data/RL
#   DATASET_NAME = WISE
#   DATASET      = ${DATA_ROOT}/${DATASET_NAME}.jsonl
#   OUTPUT       = workspace/${DATASET_NAME}/out
#
# Env overrides (same names as run.py / start_server.sh):
#   EMU_SERVER_URL     default http://127.0.0.1:23333
#   DATA_ROOT          directory holding ${DATASET_NAME}.jsonl
#   DATASET_NAME       dataset stem, e.g. WISE / factIP / gensearcher
#   IMAGE_SAVE_DIR     where the server saves decoded PNGs
#   MAX_NEW_TOKENS     per-call generation budget (default 8192)
#   MAX_WORKERS        default 1 — server-side _GEN_LOCK + vllm max_num_seqs=2
#   TEXT_TEMPERATURE   override text sampling temperature
#   TEXT_TOP_P / _K    override text top_p / top_k
#   IMAGE_TEMPERATURE / IMAGE_TOP_P / IMAGE_TOP_K  same, for image pass
#   FORCE_DRAW_ROUND   default 3 — at round N we hand-build the BoC/BoI primer
#                       and ask the server to skip text pass and just draw.

set -euo pipefail

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

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$THIS_DIR/.." && pwd)"
cd "$THIS_DIR"

: "${DATA_ROOT:=$PROJECT_ROOT/Data/RL}"
: "${DATASET_NAME:=${1:-WISE}}"

# Backward compatible forms:
#   bash run.sh factIP
#   DATASET_NAME=factIP bash run.sh
#   bash run.sh /abs/path/to/factIP.jsonl
if [[ "${1:-}" == *.jsonl || "${DATASET_NAME}" == *.jsonl ]]; then
    DATASET="${1:-$DATASET_NAME}"
    DATASET_NAME="$(basename "${DATASET%.jsonl}")"
else
    DATASET="${DATA_ROOT}/${DATASET_NAME}.jsonl"
fi

OUTPUT_DIR="${2:-$THIS_DIR/workspace/${DATASET_NAME}/out}"

if [[ ! -f "$DATASET" ]]; then
    echo "[run_agent] dataset not found: $DATASET" >&2
    echo "[run_agent] create it first, e.g.:" >&2
    echo '  echo '"'"'{"question": "draw a cat in an astronaut helmet"}'"'"' > '"$DATASET" >&2
    exit 1
fi
mkdir -p "$OUTPUT_DIR"

: "${EMU_SERVER_URL:=http://127.0.0.1:23333}"
: "${IMAGE_SAVE_DIR:=$OUTPUT_DIR/gen_images/${DATASET_NAME}}"
: "${MAX_NEW_TOKENS:=8192}"
: "${MAX_WORKERS:=1}"
: "${FORCE_DRAW_ROUND:=6}"
mkdir -p "$IMAGE_SAVE_DIR"

if ! curl -fsS -m 3 "${EMU_SERVER_URL}/health" >/dev/null 2>&1; then
    echo "[run_agent] WARN: ${EMU_SERVER_URL}/health unreachable — is start_server.sh running?" >&2
fi

DATA_DIR="$(dirname "$DATASET")"

echo "[run_agent] dataset_name=$DATASET_NAME"
echo "[run_agent] dataset=$DATASET"
echo "[run_agent] output_dir=$OUTPUT_DIR"
echo "[run_agent] image_save_dir=$IMAGE_SAVE_DIR"
echo "[run_agent] server=$EMU_SERVER_URL workers=$MAX_WORKERS"
echo "[run_agent] force_draw_round=$FORCE_DRAW_ROUND max_new_tokens=$MAX_NEW_TOKENS"

export EMU_SERVER_URL

# Optional sampling overrides — only pass them when the user set them.
extra=()
[[ -n "${TEXT_TEMPERATURE:-}" ]] && extra+=( --text_temperature "$TEXT_TEMPERATURE" )
[[ -n "${TEXT_TOP_P:-}" ]]       && extra+=( --text_top_p "$TEXT_TOP_P" )
[[ -n "${TEXT_TOP_K:-}" ]]       && extra+=( --text_top_k "$TEXT_TOP_K" )
[[ -n "${IMAGE_TEMPERATURE:-}" ]] && extra+=( --image_temperature "$IMAGE_TEMPERATURE" )
[[ -n "${IMAGE_TOP_P:-}" ]]      && extra+=( --image_top_p "$IMAGE_TOP_P" )
[[ -n "${IMAGE_TOP_K:-}" ]]      && extra+=( --image_top_k "$IMAGE_TOP_K" )

python3 -u "$THIS_DIR/run.py" \
    --model emu3p5-sft \
    --dataset "$(basename "${DATASET%.jsonl}")" \
    --output "$OUTPUT_DIR" \
    --data_dir "$DATA_DIR" \
    --server_url "$EMU_SERVER_URL" \
    --image_save_dir "$IMAGE_SAVE_DIR" \
    --max_workers "$MAX_WORKERS" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --force_draw_round "$FORCE_DRAW_ROUND" \
    "${extra[@]}"

echo
echo "[run_agent] done. Inspect:"
echo "  ${OUTPUT_DIR}/emu3p5-sft/$(basename "${DATASET%.jsonl}")/iter1.jsonl"
echo "  ${IMAGE_SAVE_DIR}/"
