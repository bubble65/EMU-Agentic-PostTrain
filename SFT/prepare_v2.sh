#!/usr/bin/env bash
# Convert rollout JSONL → tokenized SFT JSONL, parallelized across GPUs.
#
# Architecture:
#   ─ Input is line-sharded: shard i processes line_no where (line_no-1) % N == i.
#   ─ One process per GPU; each process owns one CUDA device end-to-end.
#   ─ Within a shard, image downloads are concurrent (ThreadPoolExecutor),
#     but VQ encoding stays single-threaded per GPU (no CUDA context races).
#   ─ Shards write to <output>.shard_<i> independently, then we merge in
#     line_no order so the final sft.jsonl is deterministic.
#
# Why this is safe (no image cross-talk):
#   ─ Workers are separate Python processes — url_to_label / pending_draw_call /
#     saw_draw / image cache live entirely inside one worker's address space.
#   ─ Inside a worker, image_encoder.reset_cache() runs BEFORE every sample,
#     so even though the encoder object is reused, the per-sample
#     {url -> PIL.Image} dict is wiped between samples.
#   ─ The prefetch ThreadPool only writes to per-key slots of self._pil_cache;
#     no two threads write the same key. GPU model is never touched off-main.

set -euo pipefail

# ── Configurable ──────────────────────────────────────────────────────────
PYTHON_BIN="${PYTHON_BIN:-python3}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_DIR="${DATA_DIR:-${PROJECT_DIR}/Data/SFT}"

INPUT="${INPUT:-${DATA_DIR}/UPE_raw_gensearcher_sft_trace_relpath.jsonl}"
OUTPUT="${OUTPUT:-${DATA_DIR}/converted_v2/sft.jsonl}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${PROJECT_DIR}/Emu3.5/src/tokenizer_emu3_ibq}"
VQ_PATH="${VQ_PATH:-${PROJECT_DIR}/Emu3.5-VisionTokenizer}"
TOOL_RESP_DIR="${TOOL_RESP_DIR:-${DATA_DIR}/image}"

NUM_GPUS="${NUM_GPUS:-8}"
DOWNLOAD_WORKERS="${DOWNLOAD_WORKERS:-8}"   # threads/GPU → total = NUM_GPUS * DOWNLOAD_WORKERS
DOWNLOAD_RETRIES="${DOWNLOAD_RETRIES:-3}"
DOWNLOAD_TIMEOUT="${DOWNLOAD_TIMEOUT:-45}"
LIMIT="${LIMIT:-0}"                          # 0 = all samples
KEEP_SHARDS="${KEEP_SHARDS:-1}"              # 1 = retain <output>.shard_<i>; 0 = remove after merge

LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs/prepare_v2_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${LOG_DIR}"
mkdir -p "$(dirname "${OUTPUT}")"

echo "═══════════════════════════════════════════════════════════════════"
echo "  Input:           ${INPUT}"
echo "  Output:          ${OUTPUT}"
echo "  GPUs:            ${NUM_GPUS}"
echo "  Download/GPU:    ${DOWNLOAD_WORKERS}  (total = $((NUM_GPUS * DOWNLOAD_WORKERS)))"
echo "  Download retry:  ${DOWNLOAD_RETRIES}"
echo "  Download timeout:${DOWNLOAD_TIMEOUT}s"
echo "  Limit/shard:     ${LIMIT}  (0 = all)"
echo "  Logs:            ${LOG_DIR}"
echo "═══════════════════════════════════════════════════════════════════"

# ── Launch one worker per GPU ─────────────────────────────────────────────
PIDS=()
SHARD_OUTS=()
PROGRESS_FILES=()

for ((i=0; i<NUM_GPUS; i++)); do
    SHARD_OUT="${OUTPUT}.shard_${i}"
    PROGRESS_FILE="${LOG_DIR}/shard_${i}.progress.json"
    SHARD_OUTS+=("${SHARD_OUT}")
    PROGRESS_FILES+=("${PROGRESS_FILE}")
    LOG_FILE="${LOG_DIR}/shard_${i}.log"
    echo "→ shard ${i} on cuda:${i}  →  ${SHARD_OUT}  (log: ${LOG_FILE})"

    # Pin one GPU per process via CUDA_VISIBLE_DEVICES; tell Python to use
    # cuda:0 inside that view. This is more robust than --vq-device=cuda:i
    # under some torch/driver combos.
    #
    # --quiet-progress disables tqdm so the .log file only contains real
    # events ([skip], [ERROR], the final summary). Per-shard counters go
    # to --progress-file, which the watcher aggregates onto the terminal.
    CUDA_VISIBLE_DEVICES="${i}" \
    "${PYTHON_BIN}" "${SCRIPT_DIR}/convert_v2.py" \
        --input "${INPUT}" \
        --output "${SHARD_OUT}" \
        --tokenizer-path "${TOKENIZER_PATH}" \
        --vq-path "${VQ_PATH}" \
        --vq-type ibq \
        --vq-device "cuda:0" \
        --tool-resp-dir "${TOOL_RESP_DIR}" \
        --download-workers "${DOWNLOAD_WORKERS}" \
        --download-retries "${DOWNLOAD_RETRIES}" \
        --download-timeout "${DOWNLOAD_TIMEOUT}" \
        --num-shards "${NUM_GPUS}" \
        --shard-id "${i}" \
        --limit "${LIMIT}" \
        --quiet-progress \
        --progress-file "${PROGRESS_FILE}" \
        > "${LOG_FILE}" 2>&1 &

    PIDS+=($!)
done

# ── Aggregated progress watcher (terminal only) ───────────────────────────
WATCHER_PID=""
if [[ -t 1 ]]; then
    # Only run the watcher when STDOUT is a real terminal — skip it under
    # nohup/CI/disown so we don't spam the captured log.
    "${PYTHON_BIN}" - "${LOG_DIR}" "${NUM_GPUS}" "${PIDS[@]}" <<'PY' &
import json
import os
import sys
import time

log_dir = sys.argv[1]
num_shards = int(sys.argv[2])
worker_pids = [int(p) for p in sys.argv[3:]]

def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True

def read_one(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None

paths = [os.path.join(log_dir, f"shard_{i}.progress.json") for i in range(num_shards)]
t0 = time.time()

while True:
    snaps = [read_one(p) for p in paths]
    total_planned = sum((s or {}).get("own_lines", 0) for s in snaps)
    total_done = sum((s or {}).get("total", 0) for s in snaps)
    total_written = sum((s or {}).get("written", 0) for s in snaps)
    total_skipped = sum((s or {}).get("skipped", 0) for s in snaps)
    total_imgs = sum((s or {}).get("images_encoded", 0) for s in snaps)
    elapsed = time.time() - t0
    rate = total_done / elapsed if elapsed > 0 else 0.0
    eta = (total_planned - total_done) / rate if rate > 0 else 0.0

    # One-line per-shard summary (W=written S=skipped) — short and stable
    # so the line doesn't jitter across shards of slightly different speed.
    per = " ".join(
        f"[{i}]{((snaps[i] or {}).get('total',0)):>4}/"
        f"{(snaps[i] or {}).get('own_lines','?')}"
        for i in range(num_shards)
    )
    pct = (100.0 * total_done / total_planned) if total_planned else 0.0
    bar_w = 24
    fill = int(bar_w * total_done / total_planned) if total_planned else 0
    bar = "█" * fill + "·" * (bar_w - fill)

    line = (
        f"\r[{bar}] {pct:5.1f}%  "
        f"{total_done}/{total_planned}  "
        f"W={total_written} S={total_skipped} img={total_imgs}  "
        f"{rate:5.1f}/s  ETA {int(eta//60):>3}m{int(eta%60):02d}s  | {per}"
    )
    # Pad to clear leftovers if the previous line was longer.
    sys.stdout.write(line.ljust(200)[:200])
    sys.stdout.flush()

    all_done = all((s or {}).get("done") for s in snaps) and all(s for s in snaps)
    any_workers_alive = any(alive(p) for p in worker_pids)
    if all_done or not any_workers_alive:
        sys.stdout.write("\n")
        sys.stdout.flush()
        break
    time.sleep(1.0)
PY
    WATCHER_PID=$!
fi

# ── Wait + collect exit codes ─────────────────────────────────────────────
echo
echo "Waiting for ${#PIDS[@]} workers..."
FAILED=0
for idx in "${!PIDS[@]}"; do
    pid="${PIDS[$idx]}"
    if wait "${pid}"; then
        : # success; details summarized by watcher and shard logs
    else
        rc=$?
        echo "  ✗ shard ${idx} FAILED (pid ${pid}, exit ${rc}) — see ${LOG_DIR}/shard_${idx}.log" >&2
        FAILED=1
    fi
done

# Reap the watcher cleanly.
if [[ -n "${WATCHER_PID}" ]]; then
    wait "${WATCHER_PID}" 2>/dev/null || true
fi

if [[ "${FAILED}" -ne 0 ]]; then
    echo
    echo "One or more shards failed. NOT merging. Inspect ${LOG_DIR}/." >&2
    exit 1
fi

# Print per-shard final summaries that the workers wrote at the tail of
# their own log files — much more readable now that tqdm noise is gone.
echo
echo "Per-shard final summaries:"
for ((i=0; i<NUM_GPUS; i++)); do
    echo "── shard ${i} ────────────────────────────"
    tail -n 11 "${LOG_DIR}/shard_${i}.log" | sed 's/^/  /'
done

# ── Merge shards in line_no order ─────────────────────────────────────────
echo
echo "Merging ${#SHARD_OUTS[@]} shards into ${OUTPUT} (sorted by line_no)..."

# Each shard line is a JSON object with meta.line_no. We tag each line with
# its line_no in front, sort numerically, then strip the tag. Streaming, no
# need to hold all records in memory.
"${PYTHON_BIN}" - "${OUTPUT}" "${SHARD_OUTS[@]}" <<'PY'
import json
import os
import sys
import tempfile

out_path = sys.argv[1]
shard_paths = sys.argv[2:]

# Tag every record with its line_no for a stable external sort.
tagged = tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8",
                                     prefix="sft_merge_", suffix=".tagged")
n_in = 0
for sp in shard_paths:
    if not os.path.exists(sp):
        continue
    with open(sp, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            obj = json.loads(line)
            ln = int(obj.get("meta", {}).get("line_no", 0))
            tagged.write(f"{ln}\t{line}\n")
            n_in += 1
tagged.close()

# External sort on the integer key (column 1, tab-separated).
sorted_path = tagged.name + ".sorted"
rc = os.system(f"LC_ALL=C sort -t'\t' -k1,1n -S 1G -o {sorted_path} {tagged.name}")
if rc != 0:
    sys.exit(f"sort failed with code {rc}")

n_out = 0
with open(sorted_path, "r", encoding="utf-8") as src, \
     open(out_path, "w", encoding="utf-8") as dst:
    for line in src:
        _, payload = line.split("\t", 1)
        dst.write(payload)
        n_out += 1

os.unlink(tagged.name)
os.unlink(sorted_path)
print(f"  merged {n_in} records → wrote {n_out} lines to {out_path}")
PY

# ── Optionally clean up shard files ───────────────────────────────────────
if [[ "${KEEP_SHARDS}" -eq 0 ]]; then
    for sp in "${SHARD_OUTS[@]}"; do
        [[ -f "${sp}" ]] && rm -f "${sp}"
    done
    echo "Removed shard files."
else
    echo "Kept shard files (set KEEP_SHARDS=0 to auto-delete)."
fi

echo
echo "═══════════════════════════════════════════════════════════════════"
echo "  Done.  Final output: ${OUTPUT}"
echo "═══════════════════════════════════════════════════════════════════"
