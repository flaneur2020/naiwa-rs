# M9 baseline: the binary every deletion is measured against

```
file     ~/bench-rel/bin/naiwa-rs-m9-baseline
sha256   66f8577f3e294011c47d08e6dc9eeb616eb529472a7f5d1bed1ec34a8a1b5e95
built    naiwa-rs 2f24b5a  (M9.1 part 1: L3-warming probe and ring-size knob deleted)
fork     85913adf           (fused cpu moe worker default 8, MTP head probe)
candle   59b60645           (q4k_micro)
size     274 549 200 bytes
```

Reference numbers, from this binary (counting prompt, `--depth 4`, gen 96):

| configuration | round p50 | tok/s | hit |
|---|---|---|---|
| np8, `frac 0.2`, cache 4 | **170.0 ms** | **44.0** | 87.7% |
| same + `NAIWA_NO_CAPTURE=1` (eager segments) | 170.2 ms | 44.1 | 87.1% |
| same + `NAIWA_ATTN_REGION=1` | 173.6 ms | 43.9 | 87.0% |
| same + `NAIWA_GPU_ONLY_EXPERTS=1` (lossy: perfect cache) | - | 47.4 | 100% |
| np2, `frac 0.2`, cache 4 | ~90 ms | 23-25 | ~82% |

Correctness reference: `frac 0` + `NO_LOOKAHEAD` byte-identical to `harness/m1/reference.oracle.txt`.

**Rule for the rest of M9**: after every deletion, rebuild and compare against the row above. A deletion that
moves the round by more than the run-to-run spread (~5%, i.e. outside 162-179 ms) is a regression and gets
reverted; a deletion that leaves the engine 2-5% behind is investigated before it is accepted, because it would
mean the deleted mechanism was in fact carrying something.
