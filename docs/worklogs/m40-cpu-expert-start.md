# M40 — when the CPU expert job actually starts

**Status:** reopened — closure withdrawn. See M41 for corrections. The window-average, prediction and shared-ceiling conclusions below are historical and are not valid decision evidence. **Parent:** M39 §16 (composition at `--pcie-frac 0.6`).

M39 §16 left three exposed terms at `frac 0.6`: copies 31.6 ms/token, idle 6.1, CPU wait 3.0 (arithmetic 17.4
matches Strata's 15.4). This milestone takes the one the previous session ranked first — **the CPU expert job
starts too late to be hidden** — and measures it before changing anything, because "the CPU is the critical path"
was an inference from a host-timed span (`expert_dev`), not from a boundary measurement.

## 1. The chain, named

Every layer that has misses runs this sequence on the token thread:

| # | boundary | what it means |
|---|---|---|
| B0 | top-k in host memory | the routing decision is known; nothing can be decided before it |
| B1 | staging split decided (`DmaMiss::submit` returns) | which misses go over PCIe; shrinks the CPU's set |
| B2 | CPU job submitted (`CpuMiss::submit` returns) | after a blocking `x2d` D2H (`to_device(Cpu)`) |
| B3 | worker picks the job up | executor queue delay |
| B4 | job publishes its result | the gemv time |
| B5 | GPU observes the publish | `doorbell_wait_copy_f32` — the only part of this that costs wall clock |

B0→B5 is the CPU's **window**; B4−B3 is its **cost**. The hypothesis under test is that B2−B0 (or the queue, B3−B2)
is large, so the window is short and the cost is exposed. If instead the window is short *because the GPU reaches B5
quickly*, moving the submit earlier cannot help and the only levers are (a) shortening B4 or (b) widening the window
by starting before B0 — which means speculation.

## 2. Evidence before any change (this session)

`--pcie-frac 0.6`, one engine process, one 128-token greedy request, `NAIWA_RET_DBG=1 NAIWA_ROUND_STATS=1
NAIWA_CPU_LAYER_TRACE=1` (`/tmp/step-ret06.log`):

| boundary | measurement | source |
|---|---:|---|
| B0→B2 (host, after routing is known) | **~35 µs/layer** (stage 25 µs + spawn 10 µs) | `STEP_BUDGET` stages |
| B2→B3 (executor queue delay) | **p50 10.5 µs**, p90 11.8, max 75 | `RET_DBG queue_us` (70 jobs) |
| B3→B4 (CPU gemv) | **p50 574 µs**, p90 649, min 484 | `RET_DBG compute_us` |
| B0 itself (host blocked reading top-k) | **440 µs/layer** (21.25 ms/step ÷ 48) | `STEP_BUDGET plan` |
| B5 (GPU waits for the publish) | **62 µs/layer** (3.0 ms/token ÷ 48) | nsys `doorbell_wait_copy_f32` |

Three consequences, recorded so the milestone cannot move its own goalposts:

1. **The submit is not late.** B2 comes 35 µs after the routing is known and the worker starts 10 µs after that.
   Reordering `DmaMiss`/`CpuMiss` or unblocking the `x2d` D2H is worth ≤ 45 µs/layer ≈ 2 ms/token *if the whole
   delay were exposed*, and it is not: it happens while the GPU is still executing the mixer.
2. **B0 is not a bubble.** The 440 µs the host spends blocked on the top-k read is the time the GPU needs to reach
   the router: the host arrives early and waits for the device. Idle is 6.1 ms/token (0.13 ms/layer), far below it.
3. **The entire CPU lever is 3.0 ms/token.** That is the doorbell spin — and it is also the ceiling for making the
   CPU infinitely fast, for moving the submit, and for the device-side plan (M39 §14 estimated 2.3-3.9 for the
   same money). The CPU's 12 ms of gemv/token is *not* the exposure; only its tail is.

So the interesting question is no longer "why does the CPU start late" but **"can the window be widened?"**, and the
only thing that can start work before B0 is a prediction — which already exists for the prefetch (`predict()` runs
the router on the CPU one layer ahead, §`prefetch_predicted`). If that prediction is accurate, the CPU job could be
computed one layer early and the 3.0 ms of spin would go to zero *and* `frac` could come back down (each 0.2 of
`frac` is ±4.5 ms of copy) without paying for it in CPU exposure.

That prediction's hit rate **has never been measured** — it is also what decides whether the lookahead's ~120 MB/token
of speculative H2D is paying for itself (M39 §14, open item). One probe answers both.

## 3. The probe

`NAIWA_PRED_DBG=1`, a real probe in `src/stats/predict.rs` (not a print in the hot path):

- `record_predicted(layer, ids, already_resident)` from `prefetch_predicted`, after `predict()`.
- `record_actual(layer, ids_rows)` from `Route::submit`, once the ids are on the host.

Per window of 48 layer passes it prints:

| column | meaning |
|---|---|
| `passes` | passes with both a prediction and an actual set |
| `overlap%` | Σ\|predicted ∩ actual\| / Σ\|actual\| — of the experts the layer will use, how many were predicted |
| `exact%` | passes whose predicted set equals the actual set |
| `wasted%` | Σ(predicted ids that were already resident before the prefetch) / Σ\|predicted\| — the speculative copies that could not have helped |
| `missed%` | 1 − overlap: the experts the CPU/staging path had to discover at B0 |

`overlap%` is the CPU-window lever (a speculative CPU job can only cover what was predicted), `wasted%` is the
lookahead-traffic lever, and `exact%` says whether a whole-layer speculative job is even meaningful.

## 4. Gates

| gate | requirement |
|---|---|
| oracle | `frac 0` + `--no-lookahead`, `head -1` byte-identical to `harness/m1/reference.oracle.txt` — the probe must be a no-op without `NAIWA_PRED_DBG` |
| determinism | M39's conditions still hold: three runs of one greedy request give the same text (`sha256`) at `frac > 0` |
| mechanism KPI | `PRED_DBG` overlap / wasted, plus the doorbell spin from nsys (3.0 ms/token today) |
| performance | recorded, never gated; the clean arm stays diagnostic-free |

## 5. Steps

- [x] **S1 — boundaries.** `RET_DBG` + `STEP_BUDGET` + nsys spin, table in §2. Conclusion: the submit is not late.
- [x] **S2 — the probe.** `src/stats/predict.rs` + two call sites (`prefetch_predicted`, `Route::submit`), prints
      `PRED_DBG`. Off unless `NAIWA_PRED_DBG`; the oracle is byte-identical with it compiled in (§6).
- [x] **S3 — measured** at `frac 0.6`, one 128-token request, two runs (§6).
- [x] **S4 — decided: do not build the speculative CPU job** (§7). The window cannot be widened profitably.

## 6. Result: what the prediction is worth

`--pcie-frac 0.6`, one engine process, one 128-token greedy request, `NAIWA_PRED_DBG=1` (two runs, 106 windows of
48 layer passes each; the second run has the probe's own dedupe fix):

| | mean | range across windows |
|---|---:|---|
| `overlap%` — predicted ids ∩ actual ids / actual | **64.9%** | 53-74 |
| `exact%` — passes whose predicted set equals the actual set | **2.0%** | 0-6 |
| `wasted%` — predicted ids already resident when the prefetch ran | **69.0%** | 1-100 |
| predicted / actual set size | 1.00 | — |

Three facts, and what each closes:

1. **The prediction knows about two thirds of a layer's experts, one layer early.** That is an *input* limit, not a
   router-precision limit: the lookahead predicts layer L+1 from layer L's MoE **input** (the publish happens at the
   top of the layer, before the MoE), so it never sees the activation L+1 actually routes on.
2. **It is never the exact set** (2%). So a speculative CPU job cannot be "compute the layer"; it could only be
   "compute these 10 experts and hope".
3. **In steady state 69% of what it predicts is already resident**, so the prefetch copies only the remainder. The
   ~120 MB/token that nsys saw at `frac 0.4` is therefore a *cold-cache* effect of a one-request benchmark, not a
   steady-state waste — which downgrades "cut the lookahead traffic" as a lever.

Determinism: the text at `frac 0.6` is `e19ca091…` in the clean arm and in both probe runs, so the probe is inert when
off. (The first, pre-fix probe build did flip one token — placement timing at `frac > 0` is still load-bearing, as
M39 §10 says; the shipping gate is the oracle, below.)

Oracle: `FRAC=0 MODEL=…IQ1_M NO_LOOKAHEAD=1 harness/m1/capture_oracle.sh` with the probe compiled in →
`head -1` **byte-identical** to `harness/m1/reference.oracle.txt`.

## 7. Decision: the CPU window is closed as a direction

A speculative CPU job would have to beat this arithmetic:

| term | value |
|---|---:|
| CPU gemv per layer job (p50) | 574 µs |
| layer period (46 ms / 48) | ~960 µs |
| so the CPU thread is already **60% occupied** | — |
| a speculative job over the predicted 10 experts, on top of the real 4-5 | ~3× the work = **1.7 ms/layer > the period** |
| ceiling it could win even if it were free | **3.0 ms/token** (the doorbell spin) |

The speculation costs more than it can save: the executor is a single FIFO thread, so tripling its load turns the
3.0 ms of exposure into a backed-up queue that is exposed in *every* layer. And the prize is capped at 3.0 ms/token
because that is the whole of the CPU's exposure (§2) — the 12 ms of CPU gemv is already 80% hidden.

So M40 closes with a negative result and one number to remember: **3.0 ms/token is the hard ceiling for everything
on the CPU side** (submit earlier, compute faster, speculate, or move the plan to the device — M39 §14's 2.3-3.9
estimate for the device-side plan is the same money). The remaining 46.0 → 38.2 gap is not in the CPU.

What is left, in the order §16.2 ranked it:

1. **Copies: 31.6 ms/token, ~24 exposed.** At `frac 0.6` the staged set is ~16% of selections ≈ 76 experts ≈ 140 MB
   per step, and PCIe is at 95% of its measured peak, so the only lever is fewer bytes: a larger resident set (the
   cache is 2.5 GiB of a 12 GiB card) or a better placement.
2. **Idle: 6.1 ms/token**, 58% of it immediately after a D2H — the device-side plan's slice.
3. **Arithmetic: 17.4 vs Strata's 15.4**, and M38 already showed launch count cannot win it.

Next milestone should therefore be aimed at (1), with (2) as the fallback if the cache cannot grow.
