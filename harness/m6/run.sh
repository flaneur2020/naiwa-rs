#!/usr/bin/env bash
# M6: does a captured region replayed with fresh host-written inputs match a fresh eager run?
set -euo pipefail
NVCC=${NVCC:-/usr/local/cuda/bin/nvcc}
ARCH=${ARCH:-sm_89}
"$NVCC" -O2 -arch="$ARCH" -lineinfo -o region_capture region_capture.cu
./region_capture
