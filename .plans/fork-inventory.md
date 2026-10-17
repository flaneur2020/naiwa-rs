# Fork inventory — what the mistral.rs fork carries, and what could move

Measured 2026-10-16 against the pristine upstream at `bac32a67` (the checkout in
`~/code/mistral.rs-3e28196`, same commit): **41 fork commits, 77 files, +19 448 / −152 lines.**

The delta is lopsided: `mistralrs-quant` 65 files / +18 742, `mistralrs-core` 10 files / +696.

## A. Native IQ / Q formats — `mistralrs-quant` (~14k lines) — **must stay**

`iq4nl`, `iq2s`, `iq3xxs`, `iq3s`, `q2_0` (and a large `iq4xs` expansion): each has its `.rs`,
block tables, tests, `tools/…/oracle.cpp` + testdata, the `mmq_instance_*.cu` prompt kernels, the
`mmvq_gguf.cu` vec-dot, and the `iq*_grid.cuh` tables; `gguf/ffi.rs` (+849) declares them.

They are kernel implementations plus the `GgmlDType` integration that lives in the **candle** fork.
Moving them out would mean re-exposing candle's quant types and every kernel symbol. No.

## B. CPU fused MoE — `fused_cpu.rs` 1202, `fast_mmq.rs`/`fast_mmvq.rs` +87, `cpu.rs` +296 — **must stay**

The CPU expert kernels (and the installable dispatcher the engine hooks). They belong next to the
weights and the `QTensor` they read.

## C. GPU MoE kernels — `kernels/indexed_moe.cu` +489, `kernels/moe_grouped.cu` +317 — **must stay**

The fused gate+up / down+aggregate decode kernels and the grouped GEMM; ours, but kernels.

## D. quant CUDA wiring + **our `MoeDecodeCapture`** — `gguf/cuda.rs` +701 — **stays**

~363 of those lines are `MoeDecodeCapture` (v2 M1/M2). It is *technically* movable, but it calls
five private things in the same module (`fused_gate_up_launcher`, `down_aggregate_launcher`,
`quantize_q8_1`, `q8_1_bytes`, `pad`/`MATRIX_ROW_PADDING`, `slice_ptr`). Moving it would turn those
into public API to save ~360 lines. **Not worth it** — and it belongs beside the kernels it launches.

## E. Model / pipeline plumbing — `mistralrs-core` ~40 lines — **must stay**

`moe/experts/backends.rs` (+22: route the prefill fast path by dtype — the grouped GEMM kernels do
not serve IQ4_XS, the fused decode kernels do), `pipeline/mod.rs` (+14),
`paged_attention/mod.rs` (+157: post-load budget explanation). The model calls these.

> `supports_decode_graphs` is **upstream**, not a fork change — F1 (M6) is a future edit to it.

## F. prompt-lookup speculative drafting — `speculative/` +456 — **stays**

`lookup.rs` (+364), `driver.rs` (+91), `mod.rs` (+1). Semantically an engine feature, but core's
speculative driver calls it in order; owning it in the engine would need a trait/hook **in core** —
the same fork footprint. Net zero, so it stays until v2 decides whether it keeps MTP.

## G. Measurement probes — ~150 lines spread over 9 sites — **could be deleted**

`ple.rs` `ple_gather_probe` (`PLE_DBG`), the `MOE_DBG` half of `text.rs`'s +40, and `MTP_DBG` in
`speculative/*`. These exist for the W-era measurements; the engine has its own probes
(`STEP_DBG`, the lookahead probe). Deleting them is hygiene, not a move — worth doing only once we
are sure the old numbers will not be re-measured.

## H. Already moved out — the doorbell (−1660)

`cuda/doorbell.{rs,cu}`, 14 `doorbell_*` FFI declarations, the `lib.rs` re-export. See
[`worklogs/m3-1.md`](worklogs/m3-1.md).

---

## Verdict

**The doorbell was the only clean extraction.** Everything else is either a kernel (must sit next
to its launch helpers and the candle quant types) or model plumbing (the model must be able to call
it). There is no second `doorbell`.

The only remaining shrink is **deleting the W-era probes (~150 lines)**, which is hygiene rather
than a move, and is safe to defer.
