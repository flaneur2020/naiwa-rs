#!/usr/bin/env bash
# Run the M0 harness for every arm and log each result.
#   run.sh [iters] [replays]
# m0_bakedtarget is a negative control and MUST fail: it proves the harness can
# detect a target that was baked at capture time instead of read from mapped memory.
set -euo pipefail

OUT=${OUT:-$HOME/bench-rel/naiwa-v2/m0}
ITERS=${1:-1000}
REPLAYS=${2:-200}

rc=0
for arm in m0_visibility m0_nodevfence m0_nohostfence m0_nofence; do
    echo "=== $arm (must PASS) ==="
    if "$OUT/$arm" "$ITERS" "$REPLAYS" 2>&1 | tee "$OUT/$arm.log"; then
        echo
    else
        echo "!! $arm FAILED"; rc=1
    fi
done

echo "=== m0_bakedtarget (negative control, must FAIL) ==="
if "$OUT/m0_bakedtarget" "$ITERS" "$REPLAYS" 2>&1 | tee "$OUT/m0_bakedtarget.log"; then
    echo "!! negative control PASSED -- the harness cannot detect arg baking"; rc=1
else
    echo "(expected failure confirmed)"; echo
fi
exit $rc
