#!/usr/bin/env bash
# Profile one engine under nsys (naiwa-rs, or strata with the chat config).
#
#   harness/profile/nsys_engine.sh naiwa|strata PORT TAG
#
# ncu is unavailable on this card (ERR_NVGPUCTRPERM), so nsys is the timeline tool. The
# strata arm is built from `harness/chat/engines/strata-engine.json`, patched for the
# port/log so both engines are driven by the same harness afterwards.
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
ENG=$1; PORT=$2; TAG=$3
OUT=${OUT:-$HOME/wtmp}
mkdir -p "$OUT"

if [ "$ENG" = naiwa ]; then
  MTP_N=1 harness/serve/serve.sh "$PORT" 2.2 > "$OUT"/nsys-$TAG.server.log 2>&1 &
  SRV=$!
else
  CFG="$OUT/nsys-strata-cfg.json"
  python3 - "$CFG" "$PORT" <<'PY'
import json,sys
c=json.load(open('harness/chat/engines/strata-engine.json'))
c['port']=int(sys.argv[2]); c['log']=f"{__import__('os').environ['OUT']}/nsys-strata-engine.log"; c['open_browser']=False
json.dump(c,open(sys.argv[1],'w'),indent=2)
PY
  cd "$HOME/code/strata"
  export STRATA_DECODE_TIMING=1
  ~/code/strata/.venv/bin/python serve/server.py --engine strata --config "$CFG" --port $PORT > "$OUT"/nsys-$TAG.server.log 2>&1 &
  SRV=$!
fi
for i in $(seq 1 120); do sleep 3; curl -sf --max-time 2 http://127.0.0.1:$PORT/v1/models >/dev/null && break; done
curl -sf --max-time 2 http://127.0.0.1:$PORT/v1/models >/dev/null || { echo "server $ENG never came up"; kill $SRV; exit 1; }
sleep 3
date +%s.%N > "$OUT"/nsys-$TAG.window
python3 - "$PORT" <<'PY'
import json,urllib.request,time,sys
port=sys.argv[1]
rows=[json.loads(l) for l in open("harness/chat/corpus.jsonl")]
consts=rows[0]['_constants']; d=[r for r in rows if r.get('class')=='code'][0]
msgs=[{"role":"user","content":d['turns'][0]['user'].replace('{DOC}',consts.get('DOC',''))}]
for turn in range(3):
    body={"model":"default","messages":msgs,"max_tokens":40,"temperature":0,"stream":True,
          "enable_thinking":False,"reasoning_effort":"none"}
    req=urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",data=json.dumps(body).encode(),
                              headers={"Content-Type":"application/json"})
    t0=time.time(); ttft=None; n=0; u={}
    with urllib.request.urlopen(req,timeout=1200) as r:
        for raw in r:
            line=raw.decode().strip()
            if not line.startswith("data: ") or line=="data: [DONE]": continue
            ev=json.loads(line[6:])
            if ev.get("usage"): u=ev["usage"]
            for c in ev.get("choices") or []:
                if (c.get("delta") or {}).get("content"):
                    if ttft is None: ttft=time.time()-t0
                    n+=1
    end=time.time(); ct=u.get("completion_tokens",n)
    print(f"    T{turn+1} tokens={ct} deltas={n} TPOT={(end-t0-ttft)/max(ct-1,1)*1000:.1f}ms",flush=True)
PY
date +%s.%N >> "$OUT"/nsys-$TAG.window
sleep 1
kill $SRV 2>/dev/null; sleep 3; kill -9 $SRV 2>/dev/null
