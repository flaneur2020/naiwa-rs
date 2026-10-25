# M22 — TPOT: the per-step fixed cost and the missed-expert tail

Status: **open** (planned 2026-10-25). This iteration is TPOT only; TTFT is M23.
Parent: [m21-continued-turn.md](m21-continued-turn.md) §14 (the TTFT/TPOT split), [m20-expert-cache-ceiling.md](m20-expert-cache-ceiling.md) §25, §28, §30.
One mechanism this milestone: **a per-decode-step time budget** — one instrument that splits a step's wall time into
the phases that can be changed (kernel, doorbell D2H, plan, staging DMA, CPU experts, wait), because five milestones
have now measured TPOT as a single number and every candidate mechanism has been argued from the *total*. The budget
names the phase first; only then does a mechanism get built.

---

## 1. Why this milestone exists

M21 §14 split the comparison by metric and the two gaps turned out to be unrelated:

| class | TTFT >= 2 gap | TPOT gap | which one pays |
|---|---:|---:|---|
| doc | **6.8 s** | 0.3 s | TTFT, 20x |
| short | **6.9 s** | 0.9 s | TTFT, 7x |
| long | 0.5 s | **1.7 s** | **TPOT, 3x** |
| code | 0.7 s | 0.8 s | even |

And TPOT is not one gap either. Against strata (g19, 3 rounds, KV aligned):

| class | strata p10 / p50 / p90 | naiwa p10 / p50 / p90 | reading |
|---|---|---|---|
| code | 26.9 / 29.9 / 32.7 | 32.6 / 35.7 / 37.3 | **+5.8 ms constant** |
| doc | 36.9 / 40.5 / 43.1 | 39.4 / 44.4 / 49.3 | attention side comparable, +4 ms constant |
| long | 38.5 / 39.8 / 40.7 | **38.5** / 44.1 / 49.4 | floor at parity, tail +8.7 |
| short | 39.3 / 40.9 / 42.9 | 44.4 / 50.5 / 56.1 | **+9.6 median, +13.2 tail, max 79 vs 44** |

Three facts to work from:

1. **The floor is at parity.** `long` p10 is 38.5 ms on both. When no miss lands in a step, the two decoders are the
   same speed on this card.
2. **The median loss is a constant**, not context: their TPOT rises +11 ms from a 300-row context to a 2 000-row one,
   ours +8. So attention is not it; `code`'s +5.8 ms is the term that does not depend on the conversation at all —
   48 layers x ~120 us of per-layer fixed cost.
3. **The tail is 1.4-3x**, and it is a miss inside a step (`short` max 79 ms). Strata's own distribution is 4-6 ms
   wide in every class; theirs is that tight because their misses are not on the critical path (M20 §11: their TPOT
   is flat across working sets, ours jitters with them).

So the milestone has two subjects, and they need different evidence: **the constant** (what happens in every step) and
**the tail** (what happens when a stub lands).

## 2. What has already been measured and is closed

Everything below was argued from TPOT totals, which is exactly what this milestone stops doing:

| lever | result | where |
|---|---|---|
| cache size | slope 0.4 ms/GB (was 10.7 before `--pcie-frac 0.4`); 2.3 -> 3.0 GB is worth 2.8 ms at this slope | M20 §16, §25 |
| promote-on-stage (dynamic residency) | neutral at cap 1 / min 4 sightings (-1.1 ms, inside a 6 ms spread) | M20 §25 |
| `--pcie-frac` | 0.4 is the optimum; 0.6 crosses a cliff (+6/+13 ms) | M20 §16 |
| second prefill buffer | flat curve is faster than the second buffer; not shipped | M20 §17-18 |
| borrow / shared arena | four designs rejected, and the KPI here independently rejects it | M19, M20 §15 |
| KV precision | strata's int8 vs fp16 changes 0.6 ms at most | M20 §28 |

Nothing in that list can move `short` from 50.5 to 41. The remaining candidates are the ones nobody has attributed:
**MTP acceptance** (tokens per step amortise the per-layer constant, so a low acceptance rate *is* a TPOT tax),
**lookahead depth** (`--lookahead-swaps`, prediction horizon - our existing miss-hiding), **the staging ring's shape**,
and the per-layer round trip itself.

## 3. What M22 must establish

| # | question | measurement |
|---|---|---|
| T0 | **where does one decode step's ~40 ms go?** kernel / D2H doorbell / plan / staging DMA / CPU experts / wait | the per-step budget, printed per step under a flag: phase totals and the count of steps that waited |
| T1 | how many tokens does a step actually commit, and what does acceptance cost? | MTP acceptance per step, with `--mtp-n-predict` 2/4/6 |
| T2 | does more lookahead depth pay for its copies? | `--lookahead-swaps` 2/4/8, hit rate, swaps, TPOT p50 **and p90** |
| T3 | is the tail the ring or the CPU path? | TPOT p90 against `--pcie-frac` and ring slot count, with the budget's phase split for the tail steps |

## 4. Targets

| # | target | today | strata |
|---|---|---|---|
| G1 | doc TPOT p50 | 44.4 | 40.5 |
| G2 | short TPOT p50 | 50.5 | 40.9 |
| G3 | TPOT p90, worst class | 56.1 | 42.9 |
| G4 | code TPOT p50 (the constant term) | 35.7 | 29.9 |

G1-G3 are the same statement: **close to parity on the two chat classes and stop being 1.4-3x worse in the tail.**
G4 is the per-layer constant and may be out of reach without kernel work; it is listed so the milestone says so rather
than drifting.

Non-regression: TTFT t1 / >= 2 and cache behaviour unchanged (M21's gates), oracle arm untouched, harness 0 failures,
median of >= 3 runs.

## 5. Order of work (cheapest first, each one a measurement)

1. **Build the per-step budget** (§3 T0). Without it the next four are guesses, and every previous TPOT milestone was
   argued from totals.
2. **MTP window and acceptance** (T1): tokens per step is a direct divisor of the per-layer constant. `--mtp-n-predict`
   2/4/6 with the acceptance rate in the same run.
3. **Lookahead depth** (T2/T3): `--lookahead-swaps` 2/4/8. This is the only miss-hiding mechanism we already have, and
   strata's flat TPOT says miss-hiding is where their advantage lives.
4. **Only then** a new mechanism, chosen by what the budget says: if the constant is the D2H doorbell, it is a
   pipelining problem; if it is the ring, it is a shape problem; if it is the CPU expert path, it is the kernel (and
   that is M23+/M22.2, shared with the prefill work).

## 6. Non-goals

- Anything TTFT: continued-turn chunks, turn-1 prefill, the 2.6 GB trade. Those are M23, and M21 already refuted the
  scheduling-side options.
- No cache-size experiments (closed, §2), no promotion work (closed), no borrow.
- No kernel rewrite until the budget says the kernel is the phase. If it does, that becomes M22.2 with its own
  evidence, not a guess inside this one.

## 7. Open questions

- **Is `short`'s +9.6 ms median really the constant term?** `short` has a 463-row prompt and 90 tokens per turn, and
  its TS is 51 against `code`'s 36 at a similar context - so something class-specific is happening (reuse 0.89 against
  our 0.87 on code, low). The budget should say whether `short` spends it in the CPU path or waiting.
- **What is the 48 x ~120 us?** The doorbell publishes top-k through mapped memory and the host reads it: if the host
  is polling with a lost wakeup, a step pays a fixed latency, and that would explain a *constant* that no cache change
  moves. T0 answers it.
- **Does the tail correlate with the plan's chosen ring slots?** 16 slots shared by all layers, `--pcie-frac 0.4`:
  a step that needs more misses than the ring holds waits for the ring, which is a tail with a fixed width.

## 8. First budget reading: the step is device-bound, and the host is not the problem (2026-10-24)

`NAIWA_STEP_BUDGET=1`, doc probe, one server:

```
STEP_BUDGET layers 48 total 92.9 ms | plan 84.7 spawn 1.1 preload_wait 0.0 gpu 2.7 staging_wait 3.3 cpu_join 0.4
                                    | fence 0.0 other 0.7 | waits>200us preload 0/48 staging 0/48 cpu 0/48
STEP_BUDGET layers 48 total 110.9 ms | plan 102.1 spawn 1.1 preload_wait 0.0 gpu 3.1 staging_wait 3.5 cpu_join 0.4
```

**`plan` is 91% of a decode step**, and `plan` is where the host waits for the device: `Route::submit` calls
`Doorbell::publish_and_read`, which queues the top-k ring on candle's stream and then **spins on the host** until the
device bumps the sequence counter (`wait_and_read`, `d2h/topk.rs`). So the 85-100 ms in that column is not host work -
it is the GPU doing the layer's work while the host is blocked at the doorbell. Everything the host actually does is
`spawn 1.1 + cpu_join 0.4 + staging_wait 3.3 = 4.8 ms` of a 97 ms step.

Confirmed independently: **GPU utilisation during a 99-token generation is 97% at the median, 100% at p90** (49
samples at 300 ms), memory-controller utilisation 22%. The host is idle and the device is saturated.

### What that means for the milestone

TPOT is `step_time / tokens_per_step`. Both terms are now named:

| term | measured | lever |
|---|---|---|
| step time | ~91-111 ms per pass, 48 layers x ~1.9 ms, host-blocked at each layer's doorbell | **GPU work per layer** (attention + expert kernels), and pipelining the host wait buys *nothing* while the device is at 97% |
| tokens per step | not yet measured directly; the pass count (185 passes over one session) against the tokens generated says it is small | **MTP acceptance** - the cheapest lever left, and §3's T1 |

So the prediction this milestone started with - "the constant is 48 layers x ~120 us of host round trip" - is **wrong
in the interesting direction**: the round trip is real but it is *waiting for the GPU*, so it costs what the GPU costs.
The gap to strata is therefore not host overhead either; at a similar step time it is **tokens per step**, and the rest
is per-layer kernel efficiency.

### Next, in order

1. **Measure acceptance directly** (T1): tokens committed per step, with `--mtp-n-predict` 2/4/6 and `--spec-min-p`.
   If we commit 2.2 where strata commits 3, TPOT 44 becomes 32 at the *same* step time - no kernel work at all.
2. If acceptance is already at its ceiling, the lever is the per-layer GPU pass: the budget says 1.9 ms/layer for a
   layer whose expert work is 10 experts x 3 GEMMs on a handful of rows, which is a kernel question (M22.2).
3. Do not spend anything on host-side pipelining: at 97% device utilisation there is nothing to overlap.
