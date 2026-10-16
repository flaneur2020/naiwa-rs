#!/usr/bin/env bash
# Build the M0 harness. Produces the main binary plus three negative-control arms.
set -euo pipefail

NVCC=${NVCC:-/usr/local/cuda/bin/nvcc}
ARCH=${ARCH:-sm_89}          # RTX 4070 = Ada
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${OUT:-$HOME/bench-rel/naiwa-v2/m0}
mkdir -p "$OUT"

common=(-O2 -std=c++17 -arch="$ARCH")

"$NVCC" "${common[@]}" -o "$OUT/m0_visibility" "$HERE/m0_visibility.cu"
"$NVCC" "${common[@]}" -DM0_NO_DEVICE_FENCE -o "$OUT/m0_nodevfence" "$HERE/m0_visibility.cu"
"$NVCC" "${common[@]}" -DM0_NO_HOST_FENCE -o "$OUT/m0_nohostfence" "$HERE/m0_visibility.cu"
"$NVCC" "${common[@]}" -DM0_NO_DEVICE_FENCE -DM0_NO_HOST_FENCE -o "$OUT/m0_nofence" "$HERE/m0_visibility.cu"
# Negative control: target baked at capture time. This arm MUST fail.
"$NVCC" "${common[@]}" -DM0_BAKE_TARGET -o "$OUT/m0_bakedtarget" "$HERE/m0_visibility.cu"

echo "built into $OUT (arch $ARCH)"
