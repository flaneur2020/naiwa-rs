#!/usr/bin/env bash
# Start naiwa-rs with the shipped chat configuration, for a measurement you drive yourself.
#
#   harness/serve/serve.sh [PORT] [CACHE_GB] [extra serve flags...]
#
# This is the same arm as `harness/chat/engines/naiwa.json` (MTP through the mmproj
# drafter, prefix cache on, prefill streamed) without the python driver around it: start
# it, then point `harness/tpot/tpot.py` or `harness/chat/run.py` at the port.
#
# Env overrides: BIN, FRAC (--pcie-frac), MTP_N (--mtp-n-predict), SAMPLING,
#                LOOKAHEAD_SWAPS, MIN_TOK, and MTP_FLAG="" for the no-speculation arm.
set -uo pipefail
cd "$(dirname "$0")/../.."
PORT=${1:-19980}
CACHE_GB=${2:-2.2}
shift 2 || true

MODEL=${MODEL:-$HOME/Models/qwen3-coder/IQ1_M}
FILES="Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf;Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00002-of-00002.gguf"
BIN=${BIN:-$HOME/code/naiwa-rs/target/release/naiwa-rs}
MTP_FLAG=${MTP_FLAG---mtp}

exec "$BIN" serve -m "$MODEL" -f "$FILES" \
  --cpu-moe --expert-cache-gb "$CACHE_GB" \
  --expert-profile /home/yazhou/code/strata/data/expert-profile-coder.bin \
  --pcie-frac "${FRAC:-0.4}" \
  --prefill-stream --prefill-stream-min-tokens "${MIN_TOK:-64}" \
  --max-num-batched-tokens 8192 --max-prefill-chunk-tokens 8192 --max-seqs 4 \
  $MTP_FLAG --lookahead-swaps "${LOOKAHEAD_SWAPS:-2}" \
  --mmproj "$HOME/Models/mtp/mtp-component.gguf" \
  ${MTP_N:+--mtp-n-predict "$MTP_N"} \
  --mtp-draft-sampling "${SAMPLING:-greedy}" \
  --pa-context-len 4096 --prefix-cache-n 1 \
  --host 127.0.0.1 --port "$PORT" --disable-access-log "$@"
