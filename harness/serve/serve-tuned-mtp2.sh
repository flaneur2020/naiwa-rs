#!/usr/bin/env bash
# Start naiwa-rs with the fastest verified Tuned + MTP2 profile on this 6C/12T + RTX 4070 12 GB box.
#
#   harness/serve/serve-tuned-mtp2.sh [PORT] [CACHE_GB] [extra serve flags...]
#
# Ablation result (3-arm rotated clean restarts, IQ1_M, 128 completion tokens):
#   * physical5 + CPU_BEFORE_DMA=1 + SINGLE_GPU_SAMPLE=1 + MTP2 + lookahead swaps=1 : 35.38 ms/tok (28.26 tok/s)
#   * same with --no-lookahead (pure synchronous D2D promotion)                     : 34.72 ms/tok (28.80 tok/s, 3/3 faster)
#
# Env overrides: BIN, MODEL, FRAC (0.4), MTP_N (2), SAMPLING (greedy), CONTEXT (4096),
#                MAX_SEQS (4), MIN_TOK (64), PROMOTE_MAX (1), PROMOTE_MIN_HITS (4),
#                SPIN_US (20000), CPU_THREADS (5).
set -uo pipefail
cd "$(dirname "$0")/../.."

PORT=${1:-19980}
CACHE_GB=${2:-2.5}
if [ $# -ge 2 ]; then shift 2; elif [ $# -eq 1 ]; then shift 1; fi

MODEL=${MODEL:-$HOME/Models/qwen3-coder/IQ1_M}
FILES="Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf;Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00002-of-00002.gguf"
BIN=${BIN:-$HOME/code/naiwa-rs/target/release/naiwa-rs}

export MISTRALRS_MTP_PREFIX_REPLAY=none
export MISTRALRS_ALIGN_CHECKPOINTS=1
export MISTRALRS_MOE_DET_COMBINE=1
export NAIWA_CPU_BEFORE_DMA=1
export NAIWA_SINGLE_GPU_SAMPLE=1
export NAIWA_CPU_SPIN_US=${SPIN_US:-20000}
export NAIWA_CPU_MOE_THREADS=${CPU_THREADS:-5}
export NAIWA_CPU_PHYSICAL_LAYOUT=1
export NAIWA_PROMOTE_MAX=${PROMOTE_MAX:-1}
export NAIWA_PROMOTE_MIN_HITS=${PROMOTE_MIN_HITS:-4}

exec "$BIN" serve -m "$MODEL" -f "$FILES" \
  --cpu-moe --expert-cache-gb "$CACHE_GB" \
  --expert-profile "${EXPERT_PROFILE:-$HOME/code/strata/data/expert-profile-coder.bin}" \
  --pcie-frac "${FRAC:-0.4}" \
  --prefill-stream --prefill-stream-min-tokens "${MIN_TOK:-64}" \
  --max-num-batched-tokens 8192 --max-prefill-chunk-tokens 8192 \
  --max-seqs "${MAX_SEQS:-4}" \
  --mmproj "${MMPROJ:-$HOME/Models/mtp/mtp-component.gguf}" \
  --mtp --mtp-n-predict "${MTP_N:-2}" \
  --mtp-draft-sampling "${SAMPLING:-greedy}" \
  --no-lookahead \
  --pa-context-len "${CONTEXT:-4096}" --prefix-cache-n 1 \
  --host 127.0.0.1 --port "$PORT" --disable-access-log "$@"
