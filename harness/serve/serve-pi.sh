#!/usr/bin/env bash
# Start naiwa-rs for interactive use behind a coding agent (pi / any OpenAI client).
#
#   harness/serve/serve-pi.sh [PORT]
#
# Differences from `serve.sh` (the measurement arm):
#   * 16k paged-attention context instead of 4k: a coding agent's system prompt + tool
#     declarations is ~7k tokens before the first user message, which does not fit in 4k.
#   * `--max-seqs 2` instead of 4: one conversation plus one retry; the hybrid model's
#     recurrent state is ~112 MiB per slot and the card only has ~2.4 GiB left.
#   * MTP off by default (MTP=1 turns it back on): with speculation on, a ~7k-token
#     request intermittently ends in `Invalid sampling probability at index 0: NaN`
#     (mistralrs: "the model likely produced NaN/Inf logits"). Without it the same
#     prompts and multi-turn tool calls are stable.
#
# Env overrides: CACHE_GB (2.2), MTP (0), PORT (8095).
set -uo pipefail
cd "$(dirname "$0")/../.."

MODEL=${MODEL:-$HOME/Models/qwen3-coder/IQ1_M}
FILES="Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf;Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00002-of-00002.gguf"
BIN=${BIN:-$HOME/code/naiwa-rs/target/release/naiwa-rs}
PORT=${1:-${PORT:-8095}}

ARGS=(serve -m "$MODEL" -f "$FILES"
  --cpu-moe --expert-cache-gb "${CACHE_GB:-2.2}"
  --expert-profile "${EXPERT_PROFILE:-$HOME/code/strata/data/expert-profile-coder.bin}"
  --pcie-frac "${FRAC:-0.4}"
  --prefill-stream --prefill-stream-min-tokens "${MIN_TOK:-64}"
  --max-num-batched-tokens 8192 --max-prefill-chunk-tokens 8192 --max-seqs 2
  --pa-context-len "${CONTEXT:-16384}" --prefix-cache-n 1
  --host 127.0.0.1 --port "$PORT" --disable-access-log)

if [ "${MTP:-0}" = "1" ]; then
  ARGS+=(--mtp --mmproj "${MMPROJ:-$HOME/Models/mtp/mtp-component.gguf}"
         --mtp-draft-sampling "${SAMPLING:-greedy}")
fi

export MISTRALRS_MTP_PREFIX_REPLAY=none MISTRALRS_ALIGN_CHECKPOINTS=1
exec "$BIN" "${ARGS[@]}"
