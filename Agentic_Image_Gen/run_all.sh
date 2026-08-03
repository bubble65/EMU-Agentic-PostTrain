#!/usr/bin/env bash
# Shard a JSONL dataset across the 8 replicas spawned by start_all.sh and
# run a client per replica in parallel. Each client owns 1/N of the items
# and talks to its own server, so no two clients ever contend for the same
# GPU's /generate lock.
#
# When all clients succeed, the per-shard outputs are concatenated into
# <output>/<model>/<dataset>.merged/iter*.jsonl so downstream eval can
# treat the run as a single file. Set MERGE_ON_FAIL=1 to merge partial
# results even when some shards failed (off by default — partial merges
# are easy to mistake for full ones).
#
# Usage:
#   DATASET_NAME=WISE bash run_all.sh
#   bash run_all.sh WISE
#   bash run_all.sh /path/to/dataset.jsonl
#   bash run_all.sh /path/to/dataset.jsonl /path/to/out
#
# Env (mirror run.sh; the few new ones are at the bottom):
#   DATA_ROOT         directory holding ${DATASET_NAME}.jsonl
#   DATASET_NAME      dataset stem, e.g. WISE / factIP / gensearcher
#   EMU_PORT_BASE     base port (default 23333; replica i → port BASE+i)
#   GPUS              comma list (default 0,1,2,3,4,5,6,7) — must match start_all.sh
#   IMAGE_SAVE_DIR    output PNG dir (default: <out>/gen_images)
#   MAX_NEW_TOKENS    (default 8192)
#   FORCE_DRAW_ROUND  (default 6 — same as run.sh)
#   MAX_WORKERS       per-replica thread count (default 4 — see note below)
#   TEXT_TEMPERATURE / TEXT_TOP_P / TEXT_TOP_K
#   IMAGE_TEMPERATURE / IMAGE_TOP_P / IMAGE_TOP_K
#   MERGE_ON_FAIL     1 → still merge per-shard outputs when some clients
#                     failed (default 0 — skip merge to avoid silently
#                     publishing a partial result)
#
# Note on MAX_WORKERS:
#   Per-server /generate is serialized by _GEN_LOCK. Two reasons to still
#   set MAX_WORKERS > 1 per shard: (a) /encode_images (vq encoding for
#   image_search refs) is NOT under the lock, so a second thread can be
#   preparing tool_response payloads while the first thread is generating;
#   (b) it overlaps Python-side tool dispatch (web search, image search,
#   tokenization) with GPU work. Keep it modest (default 4) so the queue
#   doesn't blow memory.

set -euo pipefail

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$THIS_DIR"

# ── Proxy (mirrors run.sh) ──────────────────────────────────────────────
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

PROJECT_ROOT="$(cd "$THIS_DIR/.." && pwd)"

: "${DATA_ROOT:=$PROJECT_ROOT/Data/RL}"
: "${DATASET_NAME:=${1:-GEN}}"

# Backward compatible forms:
#   bash runall.sh factIP
#   DATASET_NAME=factIP bash runall.sh
#   bash runall.sh /abs/path/to/factIP.jsonl
if [[ "${1:-}" == *.jsonl || "${DATASET_NAME}" == *.jsonl ]]; then
    DATASET="${1:-$DATASET_NAME}"
    DATASET_NAME="$(basename "${DATASET%.jsonl}")"
else
    DATASET="${DATA_ROOT}/${DATASET_NAME}.jsonl"
fi

OUTPUT_DIR="${2:-$THIS_DIR/workspace/${DATASET_NAME}/out}"

if [[ ! -f "$DATASET" ]]; then
    echo "[run_all] dataset not found: $DATASET" >&2
    exit 1
fi

: "${GPUS:=0,1,2,3,4,5,6,7}"
: "${EMU_PORT_BASE:=23333}"
: "${MODEL_NAME:=emu3p5-sft}"
: "${IMAGE_SAVE_DIR:=$OUTPUT_DIR/gen_images/${DATASET_NAME}}"
: "${MAX_NEW_TOKENS:=8192}"
: "${MAX_WORKERS:=4}"
: "${FORCE_DRAW_ROUND:=6}"

IFS=',' read -r -a GPU_ARR <<< "$GPUS"
N_SHARDS="${#GPU_ARR[@]}"

mkdir -p "$OUTPUT_DIR" "$IMAGE_SAVE_DIR"

WORKSPACE_DIR="$THIS_DIR/workspace/${DATASET_NAME}"
SHARD_DIR="$WORKSPACE_DIR/shards"
LOG_DIR="$WORKSPACE_DIR/logs"
mkdir -p "$SHARD_DIR" "$LOG_DIR"

DATA_BASENAME="$(basename "${DATASET%.jsonl}")"

echo "[run_all] dataset_name  = $DATASET_NAME"
echo "[run_all] dataset       = $DATASET"
echo "[run_all] output_dir    = $OUTPUT_DIR"
echo "[run_all] image_save    = $IMAGE_SAVE_DIR"
echo "[run_all] workspace     = $WORKSPACE_DIR"
echo "[run_all] shards        = $N_SHARDS (one per GPU: ${GPU_ARR[*]})"
echo "[run_all] ports         = $(seq $EMU_PORT_BASE $((EMU_PORT_BASE + N_SHARDS - 1)) | tr '\n' ' ')"
echo "[run_all] model         = $MODEL_NAME"
echo "[run_all] max_workers   = $MAX_WORKERS  per shard"
echo "[run_all] force_draw    = $FORCE_DRAW_ROUND  max_new_tokens=$MAX_NEW_TOKENS"

# ── Health check every server BEFORE we shard / launch clients ──────────
for ((i=0; i<N_SHARDS; i++)); do
    port=$((EMU_PORT_BASE + i))
    if ! curl -fsS -m 3 "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
        echo "[run_all] !! server on port $port not reachable. Is start_all.sh running and ready?" >&2
        exit 1
    fi
done

# ── Shard dataset round-robin: line i → shard (i % N_SHARDS) ────────────
# Round-robin (not block-split) so naturally uneven items — long-prompt
# blocks at the top of the file, short ones at the bottom — distribute
# evenly. Each client also re-checks `processed` from prior iter1.jsonl so
# reruns naturally skip what's already done.
echo "[run_all] sharding dataset round-robin ..."
python3 - "$DATASET" "$SHARD_DIR" "$DATA_BASENAME" "$N_SHARDS" <<'PY'
import os, sys
src, shard_dir, base, n = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
os.makedirs(shard_dir, exist_ok=True)
outs = [open(os.path.join(shard_dir, f"{base}.shard{i}.jsonl"), "w", encoding="utf-8") for i in range(n)]
counts = [0] * n
with open(src, "r", encoding="utf-8") as f:
    for idx, line in enumerate(f):
        if not line.strip():
            continue
        i = idx % n
        outs[i].write(line if line.endswith("\n") else line + "\n")
        counts[i] += 1
for o in outs:
    o.close()
print(f"[shard] split {sum(counts)} lines across {n} shards: {counts}")
PY

# ── Spawn one client per shard ──────────────────────────────────────────
PIDS=()
for ((i=0; i<N_SHARDS; i++)); do
    gpu="${GPU_ARR[$i]}"
    port=$((EMU_PORT_BASE + i))
    shard_basename="${DATA_BASENAME}.shard${i}"
    shard_file="$SHARD_DIR/${shard_basename}.jsonl"
    shard_log="$LOG_DIR/client.gpu${gpu}.log"

    # Each shard writes into the SAME OUTPUT_DIR root but a per-shard
    # `dataset` subdir, so the result file paths are deterministic and
    # don't collide: <output>/<model>/<basename>.shard<i>/iter1.jsonl
    extra=()
    [[ -n "${TEXT_TEMPERATURE:-}" ]] && extra+=( --text_temperature "$TEXT_TEMPERATURE" )
    [[ -n "${TEXT_TOP_P:-}" ]]       && extra+=( --text_top_p "$TEXT_TOP_P" )
    [[ -n "${TEXT_TOP_K:-}" ]]       && extra+=( --text_top_k "$TEXT_TOP_K" )
    [[ -n "${IMAGE_TEMPERATURE:-}" ]] && extra+=( --image_temperature "$IMAGE_TEMPERATURE" )
    [[ -n "${IMAGE_TOP_P:-}" ]]      && extra+=( --image_top_p "$IMAGE_TOP_P" )
    [[ -n "${IMAGE_TOP_K:-}" ]]      && extra+=( --image_top_k "$IMAGE_TOP_K" )

    echo "[run_all] launching shard $i  gpu=$gpu  port=$port  items=$(wc -l <"$shard_file" | tr -d ' ')  log=$shard_log"

    EMU_SERVER_URL="http://127.0.0.1:${port}" \
    python3 -u "$THIS_DIR/run.py" \
        --model "$MODEL_NAME" \
        --dataset "$shard_basename" \
        --output "$OUTPUT_DIR" \
        --data_dir "$SHARD_DIR" \
        --server_url "http://127.0.0.1:${port}" \
        --image_save_dir "$IMAGE_SAVE_DIR" \
        --max_workers "$MAX_WORKERS" \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --force_draw_round "$FORCE_DRAW_ROUND" \
        "${extra[@]}" \
        >"$shard_log" 2>&1 &
    PIDS+=("$!")
done

echo
echo "[run_all] ${#PIDS[@]} clients running. pids: ${PIDS[*]}"
echo "[run_all] tailing all client logs (Ctrl-C to stop tail; clients keep running)."

# Trap Ctrl-C → kill clients too (otherwise they'd keep hammering the servers).
cleanup() {
    echo
    echo "[run_all] stopping clients ..."
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    sleep 1
    for pid in "${PIDS[@]}"; do
        kill -9 "$pid" 2>/dev/null || true
    done
}
trap cleanup INT TERM

# Live tail of every client log.
tail -n 0 -F "$LOG_DIR"/client.gpu*.log &
TAIL_PID=$!

# Wait on all client pids. If any fail we still wait for the rest so
# partial results are flushed before we return.
FAIL=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        FAIL=$((FAIL + 1))
    fi
done

kill "$TAIL_PID" 2>/dev/null || true

echo
if (( FAIL > 0 )); then
    echo "[run_all] $FAIL/${#PIDS[@]} client(s) failed — inspect $LOG_DIR/client.gpu*.log" >&2
else
    echo "[run_all] all ${#PIDS[@]} clients finished ok."
fi

echo "[run_all] per-shard outputs under: ${OUTPUT_DIR}/${MODEL_NAME}/${DATA_BASENAME}.shard*/iter1.jsonl"
echo "[run_all] decoded PNGs:           ${IMAGE_SAVE_DIR}/"

# ── Merge per-shard outputs ─────────────────────────────────────────────
# Each shard writes <OUTPUT_DIR>/<MODEL_NAME>/<DATA_BASENAME>.shard{i}/iter{N}.jsonl.
# Concatenate same-iter files across shards in shard-index order into
# <OUTPUT_DIR>/emu3p5-sft/<DATA_BASENAME>.merged/iter{N}.jsonl so downstream
# eval doesn't have to know about the sharding.
#
# Skipped when any client failed unless MERGE_ON_FAIL=1 is set (lets you opt
# in to merging partial results — useful for debugging, dangerous otherwise).
: "${MERGE_ON_FAIL:=0}"
RESULT_ROOT="${OUTPUT_DIR}/${MODEL_NAME}"
MERGED_DIR="${RESULT_ROOT}/${DATA_BASENAME}.merged"

if (( FAIL > 0 )) && [[ "$MERGE_ON_FAIL" != "1" ]]; then
    echo "[run_all] skipping merge: $FAIL client(s) failed. Set MERGE_ON_FAIL=1 to merge partial results anyway." >&2
else
    echo
    echo "[run_all] merging per-shard outputs into: $MERGED_DIR"
    mkdir -p "$MERGED_DIR"

    # Discover the iter indices actually produced (handles --roll_out_count > 1).
    # Look in shard0 first; if a shard never produced iterN, the merge for that
    # iter just skips it with a warning.
    ITERS=()
    if [[ -d "${RESULT_ROOT}/${DATA_BASENAME}.shard0" ]]; then
        while IFS= read -r f; do
            ITERS+=("$(basename "$f" .jsonl)")
        done < <(find "${RESULT_ROOT}/${DATA_BASENAME}.shard0" -maxdepth 1 -name 'iter*.jsonl' -type f | sort)
    fi

    if (( ${#ITERS[@]} == 0 )); then
        echo "[run_all] !! no iter*.jsonl found under ${RESULT_ROOT}/${DATA_BASENAME}.shard0 — nothing to merge" >&2
    else
        for iter in "${ITERS[@]}"; do
            merged_file="$MERGED_DIR/${iter}.jsonl"
            : > "$merged_file"  # truncate; merge is idempotent w.r.t. previous runs
            total=0
            for ((i=0; i<N_SHARDS; i++)); do
                shard_file="${RESULT_ROOT}/${DATA_BASENAME}.shard${i}/${iter}.jsonl"
                if [[ ! -f "$shard_file" ]]; then
                    echo "[run_all]   shard $i missing $iter.jsonl — skipping" >&2
                    continue
                fi
                # cat preserves order within shard; shard order is i=0..N-1.
                # Tolerate files without a trailing newline (rare, but the
                # next shard would otherwise glue onto the last line).
                if [[ -s "$shard_file" ]]; then
                    cat "$shard_file" >> "$merged_file"
                    if [[ "$(tail -c1 "$shard_file" | od -An -c | tr -d ' ')" != "\\n" ]]; then
                        echo >> "$merged_file"
                    fi
                    n=$(wc -l < "$shard_file" | tr -d ' ')
                    total=$((total + n))
                fi
            done
            echo "[run_all]   $iter.jsonl: $total lines → $merged_file"
        done
        echo "[run_all] merge done."
    fi
fi

exit "$FAIL"
