#!/usr/bin/env bash
# Launch one Emu3.5 inference server per GPU (TP=1, 8 independent replicas).
#
# Why per-GPU servers instead of one TP=8 server:
#   * The /generate endpoint is serialized by _GEN_LOCK in server.py (the
#     text→image 2-pass workflow needs each pass to see exactly the prompt
#     produced by the previous pass — concurrent requests on one server
#     would interleave). With 8 replicas we get 8× throughput naturally.
#   * Each WISE prompt is independent, so dataset sharding is trivial.
#
# Each replica binds a distinct port (default 23333 + gpu_idx) and logs to
# workspace/${DATASET_NAME}/logs/server.gpuN.log. Stdout shows tail of each log.
#
# Env overrides (most pass through to start.sh, see there for full list):
#   DATASET_NAME    workspace/log namespace (default WISE)
#   GPUS            comma-separated GPU indices (default 0,1,2,3,4,5,6,7)
#   EMU_PORT_BASE   base port; replica i binds PORT_BASE+i (default 23333)
#   EMU_TP_SIZE     forced to 1 here; one replica per GPU
#   EMU_GPU_MEM_UTIL gpu mem util per replica (default 0.85 — fewer ranks
#                   sharing the device, we can claim more)
#   EMU_MODEL_PATH, EMU_VQ_PATH, EMU_TOKENIZER_PATH, ...   forwarded
#
# Run:
#   bash start_all.sh
# then in a SEPARATE shell:
#   bash run_all.sh
#
# Wait for "ready on port N" to print 8 times (one per GPU) before launching
# run_all.sh — start_all.sh blocks until either all replicas report ready,
# any replica dies, or EMU_START_TIMEOUT seconds pass (default 1800).

set -euo pipefail

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${GPUS:=0,1,2,3,4,5,6,7}"
: "${DATASET_NAME:=GEN}"
: "${EMU_PORT_BASE:=23333}"
: "${EMU_GPU_MEM_UTIL:=0.85}"
: "${EMU_START_TIMEOUT:=1800}"

WORKSPACE_DIR="$THIS_DIR/workspace/${DATASET_NAME}"
LOG_DIR="$WORKSPACE_DIR/logs"
PID_DIR="$WORKSPACE_DIR/pids"
mkdir -p "$LOG_DIR" "$PID_DIR"

IFS=',' read -r -a GPU_ARR <<< "$GPUS"
N_GPUS="${#GPU_ARR[@]}"
echo "[start_all] launching $N_GPUS servers on GPUs: ${GPU_ARR[*]}"
echo "[start_all] dataset workspace: $DATASET_NAME"
echo "[start_all] port base: $EMU_PORT_BASE  (server i → port $((EMU_PORT_BASE))+i)"
echo "[start_all] workspace: $WORKSPACE_DIR"
echo "[start_all] logs:     $LOG_DIR"

# Clean stale pid files for the ports we're about to use.
for ((i=0; i<N_GPUS; i++)); do
    rm -f "$PID_DIR/server.gpu${GPU_ARR[$i]}.pid"
done

# Spawn each replica.
PIDS=()
PORTS=()
for ((i=0; i<N_GPUS; i++)); do
    gpu="${GPU_ARR[$i]}"
    port=$((EMU_PORT_BASE + i))
    log="$LOG_DIR/server.gpu${gpu}.log"
    PORTS+=("$port")

    echo "[start_all] -> gpu=$gpu port=$port log=$log"

    # Each replica gets ONE visible GPU, so it always sees "cuda:0".
    # TP=1 → no NCCL between ranks → no contention with neighbour replicas.
    env \
        CUDA_VISIBLE_DEVICES="$gpu" \
        EMU_TP_SIZE=1 \
        EMU_PORT="$port" \
        EMU_VQ_DEVICE="cuda:0" \
        EMU_GPU_MEM_UTIL="$EMU_GPU_MEM_UTIL" \
        bash "$THIS_DIR/start.sh" \
        >"$log" 2>&1 &

    pid=$!
    PIDS+=("$pid")
    echo "$pid" > "$PID_DIR/server.gpu${gpu}.pid"
    echo "[start_all]    pid=$pid"
done

echo "[start_all] all spawned. Waiting for /health=ok on each port ..."

# Poll each server's /health until "ok", or any replica process dies, or timeout.
ELAPSED=0
SLEEP_S=5
READY=()
for _ in "${PORTS[@]}"; do READY+=(0); done

while true; do
    all_ready=1
    for ((i=0; i<N_GPUS; i++)); do
        if [[ "${READY[$i]}" == "1" ]]; then continue; fi
        port="${PORTS[$i]}"
        pid="${PIDS[$i]}"
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "[start_all] !! server pid=$pid (gpu=${GPU_ARR[$i]} port=$port) died. Tail of log:" >&2
            tail -n 50 "$LOG_DIR/server.gpu${GPU_ARR[$i]}.log" >&2 || true
            exit 1
        fi
        status="$(curl -fsS -m 2 "http://127.0.0.1:${port}/health" 2>/dev/null | tr -d '[:space:]' || true)"
        if [[ "$status" == *'"status":"ok"'* ]]; then
            echo "[start_all] ready on port $port (gpu=${GPU_ARR[$i]})"
            READY[$i]=1
        else
            all_ready=0
        fi
    done
    if [[ "$all_ready" == "1" ]]; then break; fi
    if (( ELAPSED >= EMU_START_TIMEOUT )); then
        echo "[start_all] !! timed out after ${ELAPSED}s waiting for replicas to become ready" >&2
        exit 1
    fi
    sleep "$SLEEP_S"
    ELAPSED=$((ELAPSED + SLEEP_S))
done

echo
echo "[start_all] ALL $N_GPUS servers are ready."
echo "[start_all] ports: ${PORTS[*]}"
echo "[start_all] now run:  bash $THIS_DIR/run_all.sh"
echo
echo "[start_all] tailing logs in foreground; Ctrl-C to stop AND tear down servers."

# Trap Ctrl-C → kill all replica pids cleanly.
cleanup() {
    echo
    echo "[start_all] tearing down servers ..."
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
        fi
    done
    sleep 1
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -9 "$pid" 2>/dev/null || true
        fi
    done
    echo "[start_all] bye."
}
trap cleanup INT TERM

# Tail all server logs together (prefixed). Stays in foreground so the
# process tree mirrors what `bash start.sh` would do.
tail -n 0 -F "$LOG_DIR"/server.gpu*.log
