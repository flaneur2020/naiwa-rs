#!/usr/bin/env bash
# Capture a frozen greedy reference from a naiwa-rs binary and compare against
# `harness/oracle/reference.oracle.txt`.
#
# Usage:
#   ./harness/oracle/capture_oracle.sh
#   BIN=<path> OUT=<dir> FRAC=0 NO_LOOKAHEAD=1 ./harness/oracle/capture_oracle.sh
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
BIN=${BIN:-$ROOT/target/release/naiwa-rs}
OUT=${OUT:-$HOME/wtmp/oracle}
PORT=${PORT:-12345}
PROMPT=${PROMPT:-"The quick brown fox jumps over the lazy dog. The capital of France is"}
MAXTOK=${MAXTOK:-16}
PA_CTX=${PA_CTX:-4096}
MAX_SEQS=${MAX_SEQS:-4}
MODEL=${MODEL:-$HOME/Models/qwen3-coder/IQ1_M}
FRAC=${FRAC:-0}
NO_LOOKAHEAD=${NO_LOOKAHEAD:-1}
EXTRA=${EXTRA:-}
TAG=${TAG:-$(basename "$BIN")}
if [ ! -d "$MODEL" ]; then
    echo "model directory $MODEL is missing; set MODEL= to point at a checkpoint" >&2
    exit 2
fi
mkdir -p "$OUT"
LOG="$OUT/serve.$TAG.log"

LOOKAHEAD_FLAG=""
if [ "${NO_LOOKAHEAD}" != "0" ] && [ -n "${NO_LOOKAHEAD}" ]; then
    LOOKAHEAD_FLAG="--no-lookahead"
fi

"$BIN" serve \
    -m "$MODEL" \
    -f 'Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf;Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00002-of-00002.gguf' \
    --cpu-moe --expert-cache-gb 2 --pcie-frac "$FRAC" --seed 0 \
    --pa-context-len "$PA_CTX" --prefix-cache-n 0 --max-seqs "${MAX_SEQS:-4}" \
    $LOOKAHEAD_FLAG $EXTRA \
    --host 127.0.0.1 --port "$PORT" >"$LOG" 2>&1 &
SRV=$!
trap 'kill "$SRV" 2>/dev/null || true' EXIT

for _ in $(seq 1 300); do
    curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break
    kill -0 "$SRV" 2>/dev/null || { echo "server exited; see $LOG" >&2; exit 1; }
    sleep 1
done

curl -sf "http://127.0.0.1:$PORT/v1/models" > "$OUT/models.$TAG.json"
MODEL=$(python3 -c "import json;print(json.load(open('$OUT/models.$TAG.json'))['data'][0]['id'])")

BODY=$(python3 - "$MODEL" "$PROMPT" "$MAXTOK" <<'PY'
import json, sys
model, prompt, maxtok = sys.argv[1], sys.argv[2], int(sys.argv[3])
print(json.dumps({"model": model, "prompt": prompt, "max_tokens": maxtok,
                  "temperature": 0.0, "top_p": 1.0, "seed": 0}))
PY
)
curl -sf "http://127.0.0.1:$PORT/v1/completions" -H 'Content-Type: application/json' \
    -d "$BODY" > "$OUT/reference.$TAG.json"

python3 -c "
import json
d = json.load(open('$OUT/reference.$TAG.json'))
print(repr(d['choices'][0]['text']))
" | tee "$OUT/reference.$TAG.txt"

kill "$SRV" 2>/dev/null || true
wait "$SRV" 2>/dev/null || true
echo "captured -> $OUT/reference.$TAG.txt"
