#!/usr/bin/env bash
# Run one `serve` under a 1 Hz RAM/VRAM sampler, and report the peaks with the run.
#
#   harness/bench/run_with_mem.sh TAG PORT [extra serve flags...]
#
# The roadmap asks memory/liveness claims to come from samples rather than from the
# allocator's own reporting, so the sampler is a separate loop and the log next to the
# server log is the evidence: it prints used/available RAM and used VRAM once a second,
# and the peaks are read back out of it at the end.
set -uo pipefail
cd "$(dirname "$0")/../.."
TAG=$1; shift
PORT=$1; shift
OUT=${OUT:-$HOME/wtmp}
mkdir -p "$OUT"

( while :; do
    echo "$(date +%T) ram_used_mb=$(free -m | awk '/Mem:/{print $3}') \
avail_mb=$(free -m | awk '/Mem:/{print $7}') \
vram_mb=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)"
    sleep 1
  done ) > "$OUT/mem-$TAG.log" &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null' EXIT

echo "[$(date +%T)] $TAG: $*"
start=$(date +%s)
harness/serve/serve.sh "$PORT" "${CACHE_GB:-2.2}" "$@" > "$OUT/serve-$TAG.log" 2>&1
rc=$?
echo "exit=$rc wall=$(( $(date +%s) - start ))s"
echo "peak ram_used_mb=$(grep -o 'ram_used_mb=[0-9]*' "$OUT/mem-$TAG.log" | cut -d= -f2 | sort -n | tail -1) \
min avail_mb=$(grep -o 'avail_mb=[0-9]*' "$OUT/mem-$TAG.log" | cut -d= -f2 | sort -n | head -1) \
peak vram_mb=$(grep -o 'vram_mb=[0-9]*' "$OUT/mem-$TAG.log" | cut -d= -f2 | sort -n | tail -1)"
