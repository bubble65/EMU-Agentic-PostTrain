#!/usr/bin/env bash
# Kill every running convert_v2.py worker (and the prepare_v2.sh that
# launched them, if it's still around) on THIS host.
#
# Usage:
#   bash kill_v2.sh           # list + ask before killing
#   bash kill_v2.sh -y        # skip confirmation
#   bash kill_v2.sh -9        # go straight to SIGKILL (default tries TERM
#                             # first, waits 3s, then KILL)
#   bash kill_v2.sh -y -9     # combine
#
# Matching strategy:
#   ─ Find every process whose command-line contains "convert_v2.py".
#   ─ Find every running "prepare_v2.sh" (the parent orchestrator).
#   ─ Only acts on processes owned by the current user — never touches
#     someone else's jobs.
#
# Safety:
#   ─ Refuses to kill its own PID or its parent shell.
#   ─ Prints what it's about to do and prompts unless -y is passed.

set -uo pipefail

FORCE_KILL=0   # -9 → skip TERM, go straight to KILL
ASSUME_YES=0   # -y → skip confirmation

for arg in "$@"; do
    case "${arg}" in
        -9|--force) FORCE_KILL=1 ;;
        -y|--yes)   ASSUME_YES=1 ;;
        -h|--help)
            sed -n '2,20p' "$0"
            exit 0
            ;;
        *)
            echo "unknown arg: ${arg}" >&2
            exit 2
            ;;
    esac
done

ME=$$
PARENT=$PPID
USER_NAME="$(id -un)"

# ── Collect candidate PIDs (this user only) ───────────────────────────────
# pgrep -f matches the full command line. -u limits to our uid.
mapfile -t CONVERT_PIDS < <(pgrep -u "${USER_NAME}" -f 'convert_v2\.py' 2>/dev/null || true)
mapfile -t PREPARE_PIDS < <(pgrep -u "${USER_NAME}" -f 'prepare_v2\.sh' 2>/dev/null || true)

# Filter out our own PID and our parent shell, just in case.
ALL_PIDS=()
for pid in "${CONVERT_PIDS[@]}" "${PREPARE_PIDS[@]}"; do
    [[ -z "${pid}" ]] && continue
    [[ "${pid}" == "${ME}" ]] && continue
    [[ "${pid}" == "${PARENT}" ]] && continue
    ALL_PIDS+=("${pid}")
done

# De-dup while preserving order.
declare -A SEEN=()
UNIQ_PIDS=()
for pid in "${ALL_PIDS[@]}"; do
    if [[ -z "${SEEN[$pid]:-}" ]]; then
        SEEN[$pid]=1
        UNIQ_PIDS+=("${pid}")
    fi
done

if [[ "${#UNIQ_PIDS[@]}" -eq 0 ]]; then
    echo "No running prepare_v2.sh / convert_v2.py processes found."
    exit 0
fi

# ── Show what we found ────────────────────────────────────────────────────
echo "Found ${#UNIQ_PIDS[@]} process(es):"
printf '  %-8s %-8s %-8s %s\n' "PID" "PPID" "%CPU" "CMD"
for pid in "${UNIQ_PIDS[@]}"; do
    # ps may fail if the process disappeared between pgrep and now.
    row="$(ps -o pid=,ppid=,pcpu=,args= -p "${pid}" 2>/dev/null || true)"
    [[ -z "${row}" ]] && continue
    # Truncate args column so long python invocations don't wrap badly.
    cmd_short="$(echo "${row}" | awk '{for(i=4;i<=NF;i++) printf "%s ", $i; print ""}' | cut -c1-140)"
    pid_col="$(echo "${row}" | awk '{print $1}')"
    ppid_col="$(echo "${row}" | awk '{print $2}')"
    cpu_col="$(echo "${row}" | awk '{print $3}')"
    printf '  %-8s %-8s %-8s %s\n' "${pid_col}" "${ppid_col}" "${cpu_col}" "${cmd_short}"
done

# ── Confirm ───────────────────────────────────────────────────────────────
if [[ "${ASSUME_YES}" -ne 1 ]]; then
    sig_name="TERM (then KILL after 3s)"
    [[ "${FORCE_KILL}" -eq 1 ]] && sig_name="KILL"
    read -r -p "Send SIG${sig_name} to the above? [y/N] " ans
    case "${ans}" in
        y|Y|yes|YES) ;;
        *) echo "Aborted."; exit 0 ;;
    esac
fi

# ── First pass: SIGTERM (unless -9) ───────────────────────────────────────
if [[ "${FORCE_KILL}" -ne 1 ]]; then
    echo "Sending SIGTERM..."
    for pid in "${UNIQ_PIDS[@]}"; do
        if kill -0 "${pid}" 2>/dev/null; then
            kill -TERM "${pid}" 2>/dev/null || true
        fi
    done

    # Give them up to 3s to clean up (close output files, release GPU).
    for _ in 1 2 3 4 5 6; do
        any_alive=0
        for pid in "${UNIQ_PIDS[@]}"; do
            if kill -0 "${pid}" 2>/dev/null; then
                any_alive=1
                break
            fi
        done
        [[ "${any_alive}" -eq 0 ]] && break
        sleep 0.5
    done
fi

# ── Second pass: SIGKILL anything still alive ─────────────────────────────
SURVIVORS=()
for pid in "${UNIQ_PIDS[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
        SURVIVORS+=("${pid}")
    fi
done

if [[ "${#SURVIVORS[@]}" -gt 0 ]]; then
    echo "Sending SIGKILL to ${#SURVIVORS[@]} survivor(s): ${SURVIVORS[*]}"
    for pid in "${SURVIVORS[@]}"; do
        kill -KILL "${pid}" 2>/dev/null || true
    done
    sleep 0.3
fi

# ── Final report ──────────────────────────────────────────────────────────
STILL_ALIVE=()
for pid in "${UNIQ_PIDS[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
        STILL_ALIVE+=("${pid}")
    fi
done

if [[ "${#STILL_ALIVE[@]}" -eq 0 ]]; then
    echo "✓ All targeted processes are gone."
    # Belt-and-suspenders: check again via pgrep in case new children spawned
    # during cleanup (shouldn't happen, but cheap to verify).
    REMAINING="$(pgrep -u "${USER_NAME}" -f 'convert_v2\.py|prepare_v2\.sh' 2>/dev/null | grep -v "^${ME}$" || true)"
    if [[ -n "${REMAINING}" ]]; then
        echo "  (note: new matching PIDs appeared after cleanup: ${REMAINING})" >&2
        echo "  Re-run this script to handle them." >&2
        exit 1
    fi
    exit 0
else
    echo "✗ These PIDs survived even SIGKILL (likely uninterruptible D-state on I/O):" >&2
    printf '  %s\n' "${STILL_ALIVE[@]}" >&2
    exit 1
fi
