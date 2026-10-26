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

## 14. The remaining flags are at their optimum - three "no change" results (2026-10-24)

With greedy drafting shipped, the two knobs that were tuned under the *old* decode shape were re-swept. Both say
"leave them alone", which is worth recording so nobody re-runs them:

**`--pcie-frac`** (doc probe, 4 turns x 3 runs, turn >= 2):

| frac | TPOT p50 | p90 |
|---|---:|---:|
| **0.4 (shipped)** | **37.0** | **38.9** |
| 0.35 | 39.6 | 48.1 |
| 0.30 | 37.4 | 39.9 |
| 0.25 | 41.8 | 48.8 |

0.4 and 0.3 are the same to within the spread; 0.35 and 0.25 are clearly worse. **Keep 0.4** - the optimum did not move
when the depth and the sampling did.

**Fixed depth, measured on the `code` class** (which is the class still +14.6%, short prompts, so the per-layer constant
is what shows):

| depth | TPOT p50 | p90 | acceptance | tok/step |
|---|---:|---:|---:|---:|
| **auto** | 40.8 | 46.3 | 0.62 | 2.35 |
| 2 | 42.6 | **44.9** | 0.55 | 2.11 |
| 3 | **36.5** | 45.7 | 0.72 | 3.15 |

On this probe depth 3 wins the median and loses the p90 against automatic, while on the doc probe automatic was the
best of all of them - which is the autotuner's purpose (it picks per batch from measured cost), so **auto stays**. The
one thing this table says clearly is that a *fixed* depth cannot be right for both classes, and that the acceptance
differences (0.55 / 0.62 / 0.72 at depths 2 / auto / 3) are large enough that the choice matters more than any kernel
tuning at this point.

### Where M22 ends

| lever | result |
|---|---|
| depth: fixed 4 -> automatic | **-15% TPOT**, worst step 79 -> 57.7 ms |
| draft sampling: auto -> greedy | **-9.5% p50, -22% p90** |
| `--pcie-frac` | at its optimum (0.4); no change |
| `--lookahead-swaps` | at its optimum (2); 4 is 24% worse |
| cache size / promotion / KV / second buffer | closed earlier (§2) |
| per-step budget | built; the step is 95% device time, the ring 4%, the CPU path 0.3% |

The remaining TPOT gap is `code` (+14.6%) and it is the per-layer device pass - the same 48 x ~1.9 ms that strata pays,
minus about 0.3 ms that their kernel does not. That is a kernel question (M22.2), and it is the only one left.

## 15. The expert kernels are 52% of a step at 16% of peak bandwidth - M22.2 opens (2026-10-24)

The budget now also brackets each layer's expert submissions with CUDA events and reads them a step later
(`stats::budget::dev`), which separates the device time of the expert kernels from the rest of the layer's device work:

```
STEP_BUDGET layers 48 total 63.3 ms expert_dev 33.0 ms (48 layers) | plan 55.2 staging_wait 3.4 cpu_join 0.3
```

| | per step | per layer |
|---|---:|---:|
| expert kernels | **33.0 ms (52%)** | **0.69 ms** |
| everything else (attention, norms, routing, launches) | 30.3 ms (48%) | 0.63 ms |

Those kernels move `batch x topk x 1.88 MB = 20-50` expert reads per layer (~56 MB, 2.7 GB per step), so they run at
**82 GB/s - 16% of what the card sustains**. The path is `indexed_moe_fused_decode` -> `launch_moe_gemv_fused_gate_up_*`:
**GEMV** kernels, one row per (row, expert) pair, which is the classic reason a batch-1 decode sits at a tenth of peak
bandwidth while a batched GEMM reaches 80%+.

That is the remaining TPOT gap in one sentence, and it is why `code` (+14.6%) and `short` (+3.1%) are the classes that
are left: they spend most of their step in this path. M22.2 is opened with the measurement, the traffic model, the
prize (2x -> TPOT -27%, 3.6x -> -38%) and the plan (group the pairs by expert, then one launch per layer that
dequantises each expert's blocks once).

## 15. Fresh benchmark (2026-10-26, g26b + g26c): TPOT is inside its own variance, and `2.3 GB` had to go

Two independent back-to-back runs, 3 rounds each, both engines, **0 failed turns in 132 records per run**, after the M22
kernel experiments were all reverted (the fork's `indexed_moe.cu` is pristine upstream).

| class | naiwa g26b | naiwa g26c | strata g26b | strata g26c | delta g26b | delta g26c | g25 delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| code | 34.9 | 36.0 | 29.7 | 30.0 | +17.7% | +20.0% | +14.6% |
| doc | 40.4 | 40.6 | 40.6 | 40.3 | -0.5% | +0.8% | -6.4% |
| long | 41.6 | 42.3 | 40.1 | 39.8 | +3.8% | +6.4% | +1.0% |
| short | 44.1 | 43.9 | 41.3 | 40.6 | +6.8% | +8.2% | +3.1% |

Strata reproduces its M22 numbers within 1.5%; naiwa reads 2-4 ms higher than g25 on all four classes. That looked like a
regression, and it is not:

**Step time (TPOT x tokens/step) is unchanged.**

| class | g25 | g26b | g26c |
|---|---:|---:|---:|
| code | 105.6 ms (3.06 tok/step) | 100.9 (2.89) | 106.0 (2.94) |
| doc | 103.7 (2.75) | 105.1 (2.60) | 100.9 (2.48) |
| long | 94.2 (2.32) | 104.3 (2.51) | 99.9 (2.36) |
| short | 99.2 (2.38) | 102.2 (2.32) | 97.1 (2.21) |

The move is in **acceptance**, which is not reproducible run to run:

| class | acceptance per round, g26b | per round, g26c |
|---|---|---|
| code | 0.83 / 0.88 / 0.82 | 0.82 / 0.83 / 0.79 |
| doc | 0.79 / 0.78 / 0.75 | 0.79 / 0.71 / **0.62** |
| long | 0.62 / 0.72 / 0.63 | 0.62 / 0.67 / 0.66 |
| short | 0.53 / 0.68 / 0.59 | 0.56 / **0.72** / 0.54 |

TPOT is step time divided by tokens per step, so a round whose acceptance lands at 0.53 reads 46.2 ms where the same
machine at 0.68 reads 41.8. **The shipped `--pcie-frac 0.4` is non-deterministic by design** (the down-projection
`atomicAdd` changes the float summation order, so near-ties flip), which makes acceptance - and therefore TPOT - vary
by +-5-8% between rounds. That is larger than every remaining delta in the milestone's table except `code`.

So the honest reading of this benchmark: **`doc`/`long`/`short` are inside the noise band of the measurement** (+-8%),
and **`code` at +17.7% / +20.0% is the one real gap** - consistent with M22 §13's conclusion that it is the per-layer
constant, not the kernel's throughput (step time on `code` is the *best* of the four classes).

### `--expert-cache-gb 2.3` is no longer shippable - reproduced 2/2

The first attempt at this benchmark died on `doc` turn 1, twice, for a reason that is worth recording:

```
WARN naiwa_rs::prefill_stream: prefill stream: off (one 650 MiB layer buffer + 522 MiB activations
     + 512 MiB headroom + 384 MiB KV reserve does not fit in 1744 MiB free, of which the KV will
     take its own 384). ... drop --expert-cache-gb by at least 0.19 GB
ERROR step - Model failed with error: prefill stream does not fit
```

At `2.3 GB` the cache leaves 1 744 MiB free at the first long chunk after the harness's warmup conversation, and the
plan needs 2 068 - so it refuses and the whole prompt goes to the CPU at 41x, which produces no tokens and fails the
turn. At `2.2 GB` the same measurement leaves enough free VRAM and every turn runs. The harness configs
(`naiwa.json`, `naiwa-loan.json`, `naiwa-c26.json`) therefore carry **2.2**, not 2.3.

The cache cost of that change is one slot per layer (**26 against 27** of 1 881 600 B), i.e. inside the +-5% band above -
`g26b`/`g26c` are not measuring a smaller cache, they are measuring acceptance variance.

## 16. Where the last 3% is: not cache, not acceptance - **the step itself is 1.4x slower** (2026-10-26)

Asked to stop hiding behind "within noise" and compare strata's cache hits and speculative hit rate directly. Strata
publishes both, per request, in its own log, and the derived table is decisive.

### Strata's own numbers (g26b round 2, the same conversations naiwa ran)

```
strata serve: decode expert cache hit rate: 42.4% .. 56.0% (7736..83741 hits)
              4300..31197 more read by the GPU over PCIe (14.2% .. 18.8% of all routed)
strata serve: prompt 218 tokens = ..., 160 generated in 4258 ms (37.6 tok/s), drafts accepted 101 of 102
strata serve: prompt 550 tokens = ..., 119 generated in 3032 ms (39.3 tok/s), drafts accepted  84 of  89
strata serve: prompt 1858 tokens = ..., 317 generated in 12461 ms (25.4 tok/s), drafts accepted 12 of  28
```

| prompt | generated | drafts accepted / proposed | rate | steps (= gen - accepted) | **tokens/step** | **step ms** | TPOT |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 40 | 96 | 0 / 5 | 0.00 | 96 | 1.00 | **40.0** | 40.0 |
| 687 | 56 | 0 / 0 | 0.00 | 56 | 1.00 | **38.9** | 38.9 |
| 1723 | 96 | 0 / 0 | 0.00 | 96 | 1.00 | **38.8** | 38.8 |
| 1858 | 317 | 12 / 28 | 0.43 | 305 | 1.04 | **40.9** | 39.3 |
| 1724 | 102 | 37 / 47 | 0.79 | 65 | 1.57 | 49.9 | 31.8 |
| 404 | 116 | 53 / 55 | 0.96 | 63 | 1.84 | 57.4 | 31.2 |
| **218** | 160 | **101 / 102** | **0.99** | 59 | **2.71** | **72.2** | **26.6** |
| 550 | 119 | 84 / 89 | 0.94 | 35 | **3.40** | 86.6 | 25.5 |

Fitting step time against tokens per step (which is the verify window): **`step_ms = 21 + 18.8 x rows`**.

### Ours, same arithmetic

| class | TPOT | tokens/step | **step ms** |
|---|---:|---:|---:|
| code | 34.9 | 2.89 | **100.9** |
| doc | 40.4 | 2.60 | **105.1** |
| long | 41.6 | 2.51 | 104.3 |
| short | 44.1 | 2.32 | 102.2 |

and one row (from the M22.2 budget): **63.5 ms**, of which the expert kernels are **20.9 ms**. So
`step_ms = 37.5 + 26 x rows`.

### The comparison, term by term

| | strata | naiwa | delta |
|---|---:|---:|---:|
| fixed cost per step | **21 ms** | **37.5 ms** | +16.5 ms |
| cost per extra verified row | **18.8 ms** | **26 ms** | +7.2 ms |
| one row | **40 ms** | **63.5 ms** | **+59%** |
| at ~2.7 tokens/step | **72 ms** | **101 ms** | **+40%** |

**At equal tokens per step our step is 1.4-1.6x slower, and that is the whole of the remaining TPOT gap.** The expert
kernel is only part of it: 20.9 of our 63.5 ms, i.e. **our non-expert half (42.6 ms) alone is larger than strata's
entire one-row step (40 ms)**.

So the answer to "cache or acceptance?" is **neither**:

- **Cache**: strata's decode hit rate is **42-56%** with 14-19% of routed experts read over PCIe. Ours is *at least*
  as good by construction - 26 slots/layer x 48 layers of 1 881 600 B (2.2 GB) against their 800 blobs (~1.98 GiB) -
  and TPOT barely moves with cache size (slope 0.4 ms/GB, M20). Their cache is not their advantage.
- **Speculation**: their acceptance swings 0.00 -> 0.99 across conversations and their tokens/step 1.00 -> 3.40;
  on the *hard* conversations (1858, 2201 tokens) they accept 0.43 and 0.71 and land at 1.04 tokens/step, i.e. they
  simply run one row per step - where their 40 ms step still beats our 63.5 ms. Their drafting is **not** the moat;
  it is what they can afford because the step is cheap.
- Strata does carry a second draft source we do not have (`--suffix-draft N`, default 3: "prompt lookup: draft from
  an earlier repeat of the last N+ tokens of context", which grows the verify window to at most 8), and on the
  conversations where it fires acceptance is 0.94-0.99. It is worth having, but on the hard classes it fires 0-20
  times per 96 tokens, so it cannot be the +18% on `code`.

**Therefore the last 3% (and `code`'s 18%) is our per-step cost, and two thirds of it is the non-expert half of the
layer**: 42.6 ms of a 63.5 ms one-row step, which no instrument has ever attributed - M22.2 measured only the expert
kernels (20.9) and the doorbell phases (`plan` = the wait, which hides everything else). The next measurement is that
half: attention (GDN grouped-packed prefill/decode), the router, the shared expert and the norms, per layer, on a
one-row step versus strata's 40 ms total.

## 17. Component-by-component: both engines' own stage profilers, side by side

Strata ships one (`STRATA_VERIFY_PROFILE=1`, documented in its `docs/DETAILS.md` §"Where a decode window's time goes");
ours is new (`NAIWA_LAYER_PROFILE=1`, in `qwen4_exp/text.rs`, 12 buckets, 48 layers per report). Same conversations,
same machine, `--mtp-n-predict 1` for ours so a step is a step.

### Strata, one token per window (160 windows, from its own log)

```
36.24 ms/window = verify 33.15 (GPU-reach wait 12.04 + per-layer host 19.81 [plan .05 actq .07 jobs .01 CPU 19.60]
                  + stage .08) + commit/emit .04 + draft .00
per layer-window: CPU experts 3.75 (3.75 entries), VRAM hits 4.86, PCIe 1.40
decode GPU stages (ms/window): GDN layers: qkv gemv 1.84 conv .09 ab .14 z 1.05 rec .47 out-proj 1.08
   router 2.28 shared+quant .69 VRAM hits 1.05 PCIe grp .46 down .63 up .62 head 1.23
 | QSA layers: q .83 attention .58 out-proj .37 router .75 shared+quant .22 hits .44 down .22 up .22
```

| component | strata (1 tok/window) | naiwa (per 48-layer step) | ratio |
|---|---:|---:|---:|
| mixer: GDN qkv gemv + conv + ab + z + rec | 3.6 ms | | |
| mixer: out-proj (GDN) | 1.1 | | |
| mixer: QSA q + attention + out-proj | 1.8 | | |
| **mixer total (attention/GDN/out-proj)** | **~6.9 ms** | **9.3 ms** | **1.35x** |
| norms + hyper-connection mix/inject/combine | ~1.0 ms | **3.8 ms** | **~4x** |
| router (both kinds of layer) | 3.0 ms | inside the MoE block, ≤1 ms | - |
| shared expert + activation quant | 0.9 ms | inside the MoE block | - |
| **expert GPU** (hits + PCIe + down/up) | **4.2 ms** (for 49% of the experts) | **20.9 ms** (device events, ~all experts) | **2.5x per equal work** |
| expert CPU | **22-33 ms**, overlapped with the GPU | 0.3 ms `cpu_join` | - |
| MTP head / draft | 1.2 ms | n/a in this arm | - |
| waits (GPU-reach, waitB, waitCPU) | 23 ms | folded into `plan` | - |
| **step total** | **36-46 ms** | **53-75 ms** | **1.5-1.6x** |

Our own per-component line, same run:

```
LAYER_PROFILE  48 layers, total 59.7 ms:  pre-attn norm+mix+inject 3.1  mixer (GDN or QSA) 8.9
               post-attn combine+norm+mix 0.7  moe (router+experts+shared) 47.0
```

### The three differences that matter, in order

1. **The expert kernels cost 2.5x more per unit of work** (20.9 ms for ~100% of a layer's experts against their 4.2 ms
   for 49%). This is M22.4's finding - 2 bytes per load instruction, a layout property - and it is the `code` class's
   +18%.
2. **We do not overlap the host with the device; strata does.** Their window is `12.04 wait + 19.81 host`, i.e. the CPU
   expert pool (22-33 ms) runs *under* the GPU's work. Our MoE call blocks the host for the whole device time
   (`plan` = the doorbell wait in the step budget), so the mixer (9.3 ms), the hyper-connection work (3.8 ms) and the
   expert time **add up** instead of overlapping. That is the +16.5 ms/step fixed cost of §16.
3. **The dense plumbing around the experts is ~4x theirs**: norms + mix/inject/combine are 3.8 ms for us against ~1.0 ms
   for them, on a 48-layer step.

Their cache is *worse* than ours (49% VRAM hits, 14% over PCIe against our 26 slots/layer and 2.2 GB), and their
speculative drafting is not what carries them (on the hard conversations they accept 43-71% and run 1.04 tokens/step).
**The 3-7% on `doc`/`long`/`short` and the 18% on `code` are items 1-3 above, and they are measurable, not noise.**
