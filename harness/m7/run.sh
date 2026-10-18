#!/usr/bin/env bash
set -euo pipefail
cc -O2 -march=native -pthread -o pack_layout pack_layout.c
./pack_layout
