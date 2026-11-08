# M38 — launch count on the decode critical path

**Status:** in progress (opened 2026-11-10). **Parent:** M36 (M37 closed by measurement).

M36 §15-16 closed the decode budget down to two terms: the CPU expert batch (≈19 ms/token, exposed) and everything else,
which the launch count stands for (≈70 launches per layer call, 1 433 per token). The CPU batch is hardware-bound (§11),
so the launches are the last software lever left. This milestone attacks them with **one mechanism** and measures the
result on the instrument that can actually see it.

## 1. The ceiling, measured before anything is built

Wall clock cannot see this: 70 launches per layer at a few microseconds each is inside the ±2 ms/token spread this host
has (§12). nsys can. Slicing the pure-decode window out of the existing `nsysdec-dec` capture - by taking the segments
*between* two `doorbell_wait_copy_f32_kernel` instances, since exactly one of those runs per layer call - gives:

| quantity | value |
|---|---:|
| launches per layer call | **107** (median; 70 of them inside the layer's own span) |
| inter-kernel gap per layer call | **236 µs** (median of the per-layer gap sums) |
| that gap as a fraction of a 1.6 ms layer | **15%** |
| launches per token | ~1 433 at ~20.5 layer calls per token |

So the whole launch bill is **236 µs per layer ≈ 11 ms per step ≈ 0.7 ms/token**, and that is the *entire* pot. Per
family (M36 §16): quantization 11.2 launches per layer, `gemv_kernel_batched` 6.06, cutlass bf16 GEMM 3.37,
hyper-connection 4.56, q6_k 2.27, element-wise (`badd_f32`/`affine_bf16`/`usilu_bf16`) 7.16, casts 4.13.

Two consequences, recorded up front so the milestone cannot quietly move its own goalposts:

1. **The best case is 0.7 ms/token**, achieved only by removing *every* launch. Removing the 11.2 quantisation launches,
   the largest family, is worth **24 µs per layer ≈ 1.2 ms per step ≈ 0.75 ms/token of gap**, i.e. at most that much wall.
2. **The host has slack.** The token thread spends ~1.4 ms per layer inside `Route::submit` waiting for the top-k
   (§15), and queueing 70 launches takes it ~0.2 ms. So a smaller launch count cannot speed the host up; it can only
   shorten the **GPU's** serial chain, which is what the 236 µs is. This is why the KPI below is a GPU gap, not host time.

## 2. The mechanism: emit q8_1 from the producer, not from a kernel of its own

Every quantized GEMV in the fork's `fast_mmvq` path does two launches: `quantize_q8_1` into a workspace, then the GEMV
that reads it (`mistralrs-quant/src/gguf/fast_mmvq.rs:423` and the two sibling dtype arms). The activation being
quantized was *just written* by the previous kernel - a grouped norm, `hc_mix`, or `usilu` - which already touches every
element of it.

So the mechanism is: **let the producing kernel write the q8_1 form of its output as well**, and let `fast_mmvq` consume a
pre-quantized activation instead of quantizing it. One new thing (a "produce the packed form too" variant of the kernels
that feed GEMVs, plus a `fast_mmvq` entry that takes the packed buffer), not a new scheduling idea.

Rejected alternatives, with the reason:

| alternative | why not |
|---|---|
| fuse the quantisation *into* the GEMV | the activation is re-read once per output row-block; llama.cpp keeps it separate for the same reason |
| quantise once and share across consumers | the consumers in this model are already fused (qkv, gate/up), so the activations are distinct |
| merge `usilu` + `affine` + `badd` into one element-wise kernel | worth 7.16 launches per layer but touches candle's op surface, not the fork's; a smaller, later step |
| cut casts | 4.13 per layer, each needed by a different dtype boundary |

Numerics are unchanged by construction: the q8_1 quantisation is the same arithmetic, moved to a different kernel, and the
`frac 0` oracle path (single row, CPU experts) exercises the same GEMVs.

## 3. Gates

Per the repo's rules - performance is recorded, never gated:

| gate | requirement |
|---|---|
| numeric oracle | `frac 0` + `--no-lookahead` **byte-identical** to `harness/m1/reference.oracle.txt` (`head -1`, since the script prints its own second-line prompt) |
| unit tests | `cargo test --release --features cuda --bin naiwa-rs` (56 today) and the fork's `mistralrs-quant` lib tests |
| chat harness | `harness/chat/run.py --rounds 7`, **0 empty outputs**, TPOT median recorded (40.2-40.4 ms today) |
| decode arm | 128 tokens, ≥3 rounds, median; `NAIWA_NO_X2`-style A/B arm where the mechanism has an off switch |
| mechanism KPI | nsys **gap per layer call** (236 µs today) and launches per layer call (107 today), same slicing as §1 |
| cleanup | every switch that ships stays documented; every mechanism that measures flat is deleted, not kept |

## 4. Steps

- [x] **S1 - the instrument.** A script that reproduces §1 from a capture (`/tmp/gapKPI.py`), plus the baseline in the
      table above. Needed because the wall clock cannot see any of this.
- [ ] **S2 - packed activation buffer.** The producer-side variant for the single highest-count consumer, chosen by
      counting the `fast_mmvq` call sites per layer rather than guessing. Correctness first: same bytes as
      `quantize_q8_1`, checked against it on random activations.
- [ ] **S3 - wire the rest of the feed-forward path** once S2 shows the gap moving.
- [ ] **S4 - measure**: gap per layer, launches per layer, decode median over ≥3 rounds, chat harness, oracle.
- [ ] **S5 - decide**: keep only what the gap instrument can see; delete the rest.

## 5. Evidence

| when | what | result |
|---|---|---|
| 2026-11-10 | ceiling from `nsysdec-dec` | 107 launches and **236 µs** of gaps per layer call; quantisation is 11.2 of the launches |

### S1 evidence (2026-11-10)

`harness/m38/gap_kpi.py` reproduces §1 from any decode capture: it slices the segments between two
`doorbell_wait_copy_f32_kernel` instances and reports launches and gap sums per layer call. On `nsysdec-dec`:
2 626 layer calls, **107 launches median (125 mean)**, **236 µs of gaps median**, 15% of a 1.6 ms layer.

The counting half of S1 is a probe in the fork: `NAIWA_MMVQ_DBG=1` makes `fast_mmvq::plain` keep a histogram of
`(dtype, k, nrows, rows)` and print the top call sites every 2 000 calls. One 64-token decode run:

| projection | MMAVQ call shape | share of calls |
|---|---|---:|
| HC `up` (lowrank -> hc x hidden) | IQ4NL k=640 nrows=2560 | ~15% across row counts |
| attention QKV | Q6K k=2560 nrows=6144 | ~13% |
| HC `down` (hc x hidden -> lowrank) | IQ4XS / Q4K k=2560 nrows=640 | ~8% |
| MLP gate+up | Q5K / Q6K k=2560 nrows=10240 | ~3-4% |
| attention out | Q6K k=6144 nrows=2560 | ~11% |

So the ~8 quantisations per layer are **one per projection, no sharing** - the consumers are already fused (qkv, gate/up),
which is why §2's "share the packed activation" alternative is dead and the mechanism has to be producer-side.

The producers split cleanly, and only two of them are ours: `hc_mix` and `hc_combine` are fork-side kernels
(`mistralrs-core/src/cuda/qwen4_exp`), while the `silu` that feeds the HC `up` GEMV is candle's `usilu_bf16`, outside the
fork's edit surface. **S2 therefore starts with `hc_mix` -> attention QKV and `hc_combine` -> HC `down`: 2 of the 11.2
launches per layer, i.e. ~4.4 µs of gap per layer, ~0.1 ms/token.** That is small on its own, and S2 exists to prove the
pipeline end to end; the rest of the count is behind candle-side element-wise kernels and is deferred to S3.
