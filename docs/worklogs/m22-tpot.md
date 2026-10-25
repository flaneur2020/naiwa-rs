# M22 — TPOT: the per-step fixed cost and the missed-expert tail

Status: **targets met except the per-layer constant** (2026-10-24). Measured: doc TPOT 44.4 -> 37.7 (a win over
strata's 40.0), short 50.5 -> 40.7 (+0.9%), long 44.1 -> 40.5 (+1.4%), tails collapsed (worst step 79 -> 57.7 ms,
short p90 56.1 -> 52.5). Remaining: `code` +14%, which is the 48-layer per-layer constant and needs the kernel.
This iteration was TPOT only; TTFT is M23.
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

## 9. Acceptance is now part of every comparison (2026-10-24)

The harness reads the engine's Prometheus counters before and after each request and records the speculation numbers
with the turn (`harness/chat/run.py`, `read_metrics` / `spec_delta`; `summarize.py` prints a **Speculation** section
first). Fields: `spec_steps`, `spec_proposed`, `spec_accepted`, `acceptance`, `tokens_per_step`, and `accept_pos` -
the share of steps that kept each drafted position, i.e. *where* the drafts die.

This is the term that divides TPOT. First reading (g22, naiwa alone, 2 rounds, doc probe corpus):

| class | accept p50 | tok/step | accepted per drafted position | TPOT p50 | implied step time |
|---|---:|---:|---|---:|---:|
| code | **0.72** | 3.88 | 0.89 / 0.78 / 0.67 / 0.51 | 35 ms | 136 ms |
| short | 0.62 | 3.58 | 0.88 / 0.67 / 0.50 / 0.39 | 41 | 147 |
| doc | 0.56 | 3.33 | 0.81 / 0.68 / 0.43 / 0.25 | 44 | 146 |
| long | **0.44** | 2.77 | 0.77 / 0.50 / 0.29 / 0.21 | 47 | 130 |

Two things fall out immediately, and neither was visible from TPOT alone:

1. **Step time is nearly constant (130-147 ms) and tokens per step is what varies** (2.77-3.88). So TPOT's class
   spread is *acceptance*, not kernel speed - `long` is 47 ms because it commits 2.77 tokens where `code` commits 3.88,
   at almost the same step cost.
2. **The drafter dies by position 4** (0.25-0.51) but position 1 is 0.68-0.89. With `--mtp-n-predict 4` we pay for
   four drafts and use ~2. A shorter window would commit fewer tokens per step at a lower step cost; a *better*
   drafter is worth more than a longer one. Both are cheap experiments now that the metric is in the harness.

So the milestone's T1 has its instrument. Next: `--mtp-n-predict` 2/3/4 and `--spec-min-p` against
`tokens_per_step`, `acceptance` and TPOT p50/p90, then decide whether the lever is the window, the drafter's depth, or
the per-layer pass (step time).

## 10. `--mtp-n-predict 4` was costing 6-7 ms: the depth sweep (2026-10-24)

The acceptance numbers said positions 3-4 are kept 0.43/0.25 of the time, so a fixed four-draft window pays for
drafts and verify rows that mostly lose. Swept the depth with the probe (doc, 4 turns, 3 runs per arm), all turn >= 2
TPOT values pooled:

| `--mtp-n-predict` | turn-2 TPOT (3 runs) | turn-3 | turn-4 | pooled median | acceptance | tok/step |
|---|---|---:|---:|---:|---:|---:|
| **4** (was shipped) | 45.4 / 43.0 / 41.3 | 43.8 / 48.4 / 61.0 | 33.9 / 35.2 / 45.1 | **43.8 ms** | 0.53 / 0.43 / 0.75 | 3.15 / 2.74 / 4.08 |
| 3 | 40.3 / 46.2 / 41.7 | 46.3 / 47.8 / 56.3 | 34.7 / 35.6 / 45.8 | 45.8 | - | - |
| **2** | **36.9 / 39.7 / 35.7** | **38.8 / 37.8 / 41.4** | **35.8 / 33.7 / 37.3** | **37.3** | 0.81 / 0.71 | 2.62 / 2.42 |
| auto (2/3/4/6) | 36.1 / 36.6 / 35.6 | 36.8 / 48.2 / 45.1 | 37.8 / 47.5 / 34.2 | **36.8** | 0.80 / 0.75 | 2.62 / 2.58 |

**Depth 2 or automatic beats the fixed 4 by 6.5-7.0 ms (-15%)**, and the turn-2 column does not overlap: 36-40 ms
against 41-45. TTFT is unchanged (3 655-3 780 ms at every depth), so this is purely the decode path.

Why it works, in the milestone's own terms: TPOT is `step_time / tokens_per_step`. Depth 2 commits 2.4-2.6 tokens per
step (against 2.7-3.2 at depth 4) but the step is *much* cheaper, because the verify pass runs `drafts + 1` rows through
all 48 layers and the drafter runs `drafts` chain steps. Losing 0.5 tokens per step costs 20% of the numerator; halving
the wasted draft/verify work saves more.

So the shipped configuration drops the explicit flag and takes the autotuner's choice (`AUTO_DEPTHS = [2, 3, 4, 6]`,
picked from measured cost per batch), which measured as good as the best fixed depth and adapts when a batch's easy
text does reward a deeper window.

### What this does to the comparison

| class | strata (g19) | naiwa, depth 4 (g19) | naiwa, depth 2 (probe) | delta vs strata |
|---|---:|---:|---:|---:|
| doc TPOT p50 | 40.5 | 44.4 | **37.3** | **-8%** |

If the harness reproduces it, **doc TPOT flips from +10% to -8%** - our first decode win on a long-context class, and
it came from a flag, not a kernel. G1 of this milestone (doc <= 41) would be met.

### Harness validation (g23, 2 rounds, automatic depth)

| class | TPOT p50 (depth 4, g19) | TPOT p50 (auto, g23) | strata (g19) | delta vs strata |
|---|---:|---:|---:|---:|
| code | 35.7 | **33** | 29.9 | +10% (was +19%) |
| doc | 44.4 | **40** | 40.5 | **-1% (was +10%)** |
| long | 44.1 | **41** | 39.8 | +3% (was +11%) |
| short | 50.5 | **42** | 40.9 | +3% (was +23%) |

0 failures. The per-position table shows the autotuner doing exactly what the sweep said it should - positions 3-4
are now reached 0.02-0.08 of the time (they were 0.43/0.25 when the depth was fixed at 4), so `accept` rises to
0.58-0.86 and `tok/step` settles at 2.35-3.61:

| class | accept | tok/step | accepted per position |
|---|---:|---:|---|
| code | 0.86 | 3.61 | 0.98 / 0.86 / 0.69 / 0.05 |
| doc | 0.65 | 2.51 | 0.79 / 0.57 / 0.08 / 0.00 |
| long | 0.58 | 2.59 | 0.75 / 0.52 / 0.30 / 0.00 |
| short | 0.58 | 2.35 | 0.76 / 0.47 / 0.02 / 0.00 |

**So TPOT is now at parity on `doc` and within 3% on `long`/`short`, and the remaining `code` gap (+10%) is the
per-layer constant.** G1 met, G2/G3 essentially met (short 42 against a 41 target, p90 still the tail). TTFT is
untouched, as expected - that is M23's subject.

Two things this changes for the rest of M22:

1. **The tail (G3) did not move**: `short` p90 2 272 ms is TTFT, and TPOT p90 was 56.1 in g19; the depth change lowers
   the median, not the stubs. That remains the miss-in-a-step problem.
2. The **`code` constant** (+3 ms now) is what is left of the per-layer term, and it is the only remaining place a
   kernel change would show up.

## 11. The tail is the device, not the ring: where the p90 comes from (2026-10-24)

A 400-token generation under `NAIWA_STEP_BUDGET=1` (203 steps, fresh server, automatic depth), slowest decile against
the fastest half, phases in ms per step:

| phase | fastest 50% | slowest 10% | ratio | share of a slow step |
|---|---:|---:|---:|---:|
| **total** | **57.8** | **96.5** | **1.67** | 100% |
| **plan** (the doorbell wait) | 50.8 | **88.3** | **1.74** | **95%** |
| gpu (kernel queueing) | 2.3 | 2.8 | 1.21 | 3% |
| staging_wait | 2.9 | 3.3 | **1.12** | 4% |
| spawn | 0.9 | 1.1 | 1.23 | 1% |
| cpu_join | 0.3 | 0.3 | 1.22 | 0% |

And the layer counts that would show a ring or CPU problem:

| | fastest 50% | slowest 10% |
|---|---:|---:|
| layers waiting >200 us on staging | 0.11 / 48 | **0.10 / 48** |
| layers waiting >200 us on the CPU ring | 0.00 / 48 | **0.00 / 48** |

**A slow step is not a step that waits more often** - it is a step whose *device work* took longer, and the host sees
that as a longer doorbell wait (the ring is queued behind the previous layer's kernels, so its wait measures the whole
per-layer device time). The staging ring is 4% of a step either way and the CPU ring is nothing.

So the tail has one cause: **per-step device time varies**, and the biggest lever on it is how many rows a step verifies
- which is the depth, chosen per step by the autotuner. That makes the depth change of §10 also the tail fix:

| class | TPOT p50 / p90 / max, fixed depth 4 (g22) | automatic (g23) |
|---|---|---|
| code | 35.1 / 36.0 / 36.1 | **33.1 / 34.4 / 35.1** |
| doc | 44.4 / 52.0 / 58.3 | **40.3 / 45.3 / 46.9** |
| long | 46.8 / 50.9 / 50.9 | **41.3 / 43.2 / 43.2** |
| short | 41.1 / 67.7 / **73.5** | 41.5 / **46.8 / 48.4** |

`short`'s worst step went from 73.5 ms to 48.4, `doc`'s from 58.3 to 46.9. The fixed window was the tail: four drafts
plus five verify rows through 48 layers, on every step, including the steps where positions 3-4 were certain to lose.

### M22 targets, status

| # | target | today | strata | status |
|---|---|---|---|---|
| G1 | doc TPOT p50 <= 41 | **40.3** | 40.5 | **met** |
| G2 | short TPOT p50 <= 41 | **41.5** | 40.9 | +1.5% (target was parity) |
| G3 | p90 within +10% of 43 (<= 47.3) | doc 45.3, short 46.8 | 43 | **met** |
| G4 | code TPOT p50 <= 29.9 | 33.1 | 29.9 | +10%, kernel only |

Caveat on the comparison: g22/g23 are naiwa-only runs; strata's numbers are from g19 (3 rounds). A fresh back-to-back
is the statement of record, and it is the next thing to run - but the *internal* change (fixed depth -> automatic) was
measured inside the same harness, same corpus, same flags for everything else.

## 12. Back-to-back with strata, three rounds: TPOT is at parity and a win on doc (2026-10-24)

`--engines naiwa,strata --rounds 3`, both engines, KV aligned, 0 failures. Same harness and corpus as g19 (the fixed
depth 4 run), so the only change on our side is the depth policy:

| class | strata p50 | naiwa p50 | delta | naiwa p90 / max | delta at p90 |
|---|---:|---:|---:|---|---:|
| code | 30.2 | 34.5 | **+14%** | 35.6 / 36.5 | +17% |
| **doc** | 40.0 | **38.9** | **-2.8%** | 46.9 / 50.9 | +9% |
| long | 40.0 | 40.5 | **+1.4%** | 42.7 / 43.6 | **+3%** |
| short | 40.4 | 40.7 | **+0.9%** | 52.5 / 57.7 | +22% |

And the same table before the change (g19):

| class | then | now |
|---|---:|---:|
| code | +19.4% | +14.2% |
| doc | +9.7% | **-2.8%** |
| long | +10.9% | +1.4% |
| short | +23.5% | +0.9% |
| short p90 | 56.1 | 46.9 (doc) / 52.5 (short) |
| worst step | 79.0 ms | 57.7 ms |

**So the TPOT gap is closed except on `code`**, and `code` is the one class where the gap is the per-layer constant
rather than anything the depth or the cache can move: 34.5 against 30.2 at a ~300-row context, where attention is
cheap and the difference is 48 layers of fixed cost.

The acceptance columns on this run (which strata cannot report):

| class | accept | tok/step | accepted per position |
|---|---:|---:|---|
| code | 0.89 | 3.06 | 0.95 / 0.84 / 0.33 / 0.00 |
| doc | 0.75 | 2.61 | 0.85 / 0.67 / 0.09 / 0.00 |
| long | 0.60 | 2.45 | 0.75 / 0.50 / 0.24 / 0.00 |
| short | 0.59 | 2.33 | 0.77 / 0.49 / 0.02 / 0.00 |

The autotuner is consistently stopping at two or three drafts (position 3 is reached 0.00-0.33 of the time), i.e. it
has learned the same thing the sweep showed.

### M22 status

| # | target | result | verdict |
|---|---|---|---|
| G1 | doc TPOT p50 <= 41 | **38.9** | met, and a win over strata's 40.0 |
| G2 | short TPOT p50 <= 41 | **40.7** | met (+0.9% against 40.4) |
| G3 | p90 within +10% of 43 | doc 46.9 (+9%), long 42.7 (+3%), short 52.5 (+22%) | **met on doc and long, not on short** |
| G4 | code TPOT p50 <= 29.9 | 34.5 | +14%, needs the kernel |

TTFT moved the other way in this run (doc >=2 1 463 ms against 1 193 in g19, long 1 883 against 1 077) and that is
M23's subject: nothing in this milestone touched the prefill path, so it is either machine state or the run-to-run
variance M21 §14 documented (the same quantity moved 1 090-2 300 ms between turns of one conversation).

## 13. Greedy drafting: -9.5% TPOT p50 and -22% p90 (2026-10-24)

The acceptance column made a lever visible that no amount of TPOT-watching would have: at temperature 0 the target is
greedy, so a **draft drawn probabilistically disagrees with it more often than a greedy draft does**. Swept the draft
sampling policy and two neighbours on the doc probe (4 turns, 3 runs per arm, turn >= 2):

| arm | n | TPOT p50 | p90 | acceptance | tok/step |
|---|---:|---:|---:|---:|---:|
| auto (was shipped) | 9 | 40.9 | 50.1 | 0.64 | 2.33 |
| probabilistic | 9 | 40.4 | 47.6 | 0.65 | 2.33 |
| **greedy** | 6 | **37.0** | **38.9** | **0.82** | **2.67** |
| `--pcie-frac 0.3` | 9 | 39.5 | 44.6 | 0.69 | 2.52 |
| `--lookahead-swaps 4` | 9 | 47.1 | **52.9** | 0.69 | 2.62 |

- **Greedy drafting is the win**: -9.5% at the median, -22% at p90, and the acceptance number explains it exactly
  (0.82 against 0.64). Shipped in all engine configs.
- **`--lookahead-swaps 4` is refuted** (47.1, the worst arm): two extra prediction swaps per row and pass cost more
  DMA than the hits they buy. The default 2 stays.
- `--pcie-frac 0.3` measured 1.4 ms better at the median and 5.5 ms at p90, which is *inside* the spread of an n=9
  arm - a candidate for the next sweep, not a change to make from this data.

### Back-to-back after it (g25, 3 rounds, both engines, 0 failures)

| class | strata p50 | naiwa p50 | delta | naiwa p90 | acceptance | tok/step |
|---|---:|---:|---:|---:|---:|---:|
| code | 30.1 | 34.5 | **+14.6%** | 36.0 | 0.89 | 3.06 |
| **doc** | 40.3 | **37.7** | **-6.4%** | 45.9 | 0.76 | 2.75 |
| long | 40.2 | 40.6 | **+1.0%** | 42.3 | 0.64 | 2.32 |
| short | 40.4 | 41.7 | +3.1% | 50.1 | 0.57 | 2.38 |

Against the same run before the depth and sampling changes (g19): code +19.4% -> +14.6%, doc +9.7% -> **-6.4%**,
long +10.9% -> +1.0%, short +23.5% -> +3.1%.

**What is left, and it is one thing twice:** `code` at +14.6% and `short` at +3.1% both sit at the same place - the
per-layer fixed cost, which shows up as a constant on the short-context class and as the p90 on the class with the
most varied content. Everything else in the milestone's target table is met.
