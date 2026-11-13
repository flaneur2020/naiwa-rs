#!/usr/bin/env bash
# Same expert partition, only resident submission order changes. No drafting:
# --mtp-n-predict cannot be supplied without --mtp (clap requires it).
set -euo pipefail
cd /home/yazhou/code/naiwa-rs
for round in 1 2 3; do
    modes='base early'
    if [[ $round == 2 ]]; then modes='early base'; fi
    for mode in $modes; do
        echo "ARM round=$round mode=$mode"
        first=0; [[ $mode == early ]] && first=1
        env -u NAIWA_CPU_ACT_ASYNC -u NAIWA_STAGING_WAVES \
            NAIWA_RESIDENT_FIRST=$first NAIWA_CPU_PHYSICAL_LAYOUT=1 \
            NAIWA_CPU_MOE_THREADS=5 NAIWA_CPU_SPIN_US=20000 \
            MISTRALRS_MOE_DET_COMBINE=1 \
            bash /tmp/m41-nospec-naiwa.sh 0.25 m42-order-$mode-r$round
        # No request retries. Every process must have produced all three valid captures.
        python3 - "$mode" "$round" <<'PY'
import json, sys
for i in (1,2,3):
    p=f'/tmp/pf-m42-order-{sys.argv[1]}-r{sys.argv[2]}-{i}.json'
    d=json.load(open(p))
    assert d['usage']['completion_tokens']==128, (p,d['usage'])
PY
    done
done
