#!/usr/bin/env bash
# Capture a frozen greedy reference from a naiwa-rs binary, and (later) the same capture from
# the v2 binary so the two can be diffed token for token.
#
# Usage:  BIN=<path> OUT=<dir> harness/m1/capture_oracle.sh
#   BIN  defaults to the qflash-rs oracle binary
#   OUT  defaults to ~/bench-rel/naiwa-v2/m1
set -euo pipefail

BIN=${BIN:-$HOME/code/qflash-rs/target/release/naiwa-rs}
OUT=${OUT:-$HOME/bench-rel/naiwa-v2/m1}
PORT=${PORT:-12345}
PROMPT=${PROMPT:-"The quick brown fox jumps over the lazy dog. The capital of France is"}
MAXTOK=${MAXTOK:-16}
# A small KV budget: the server's auto-size reserving runtime memory too squeezed the KV pool
# out on this 12 GB card. Context length does not affect a 16-token greedy decode.
PA_CTX=${PA_CTX:-4096}
TAG=${TAG:-$(basename "$BIN")}
mkdir -p "$OUT"
LOG="$OUT/serve.$TAG.log"

"$BIN" serve \
    -m "$HOME/models/qwen3-coder/requant" \
    -f 'Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf;Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00002-of-00002.gguf' \
    --cpu-moe --expert-cache-gb 2 --pcie-frac 0.55 --seed 0 \
    --pa-context-len "$PA_CTX" --prefix-cache-n 0 \
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
