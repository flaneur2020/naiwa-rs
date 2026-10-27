# M27 - expert-cache hit rate (the numerator)

**Status:** open, **research only - nothing built**. This is the last lever on the TPOT ledger: M24 showed the decode is
transfer-bound, M25 closed the denominator (no second drafter helps), M26 closed reordering (the copy engine is already
71% busy). What is left is **fewer bytes per step**, and this milestone asks where the misses come from.
**Parent:** M24 (§7: 394 MB per step, 71% of the link), M25 (closed negative), M26 (closed negative), M20 (the 2.2 GB
ceiling that this milestone must live inside).

## 1. The prize, in numbers

Per step (2 rows x topk 10 x 48 layers), from `STEP_BUDGET` and the copy-engine accounting:

| quantity | value | how |
|---|---:|---|
| touches per layer per step | **20** | 2 rows x topk 10 |
| distinct experts touched | **15.4** | `distinct 739` of `pairs 960` |
| **misses** | **4.4** | 394 MB/step ÷ 1.88 MB ÷ 48 |
| hits of touches | **78%** | (20 - 4.4)/20 |
| resident share of distinct | **71%** | (15.4 - 4.4)/15.4 |
| slots | **26 per layer, 1 248 total, 2.35 GB** | M20's ceiling, shipped as 2.2 |

Strata runs at a **~13% miss** (its log: "+12.9% of the routed experts over PCIe"). Closing the 22% -> 13% gap is
**1.8 more hits per layer per step**, i.e. **162 MB less per step**, i.e. **12.7 ms of copy-engine time**:

> **TPOT -16%** (step 80 ms -> 67 ms) for no change in model, memory or accuracy.

That is the largest single number left on the ledger, which is why it deserves its own milestone.

## 2. What the current policy is, and why it is already good

- Slots are **fixed per layer** (26 each) and **pre-filled from `--expert-profile`** (strata's ranking, best first), so
  the cache starts holding each layer's top-26 by construction.
- On a miss the expert is **staged** (DMA into the staging ring) and **promoted** at `NAIWA_PROMOTE_MAX = 1` per layer
  per step, gated by `NAIWA_PROMOTE_MIN_HITS = 4` demand sightings - a "second sighting, at least" admission rule that
  exists because an aggressive version (cap 8, promote anything) *lost 2-4 ms* by evicting hot residents for rare ones.
- With a 26-of-256 cache, a uniform router would miss ~90% of touches. We miss **22%**. The skew is real and the
  profile pinning already collects most of it: the top-26 set covers ~78% of touches (14% under the current reading of
  "coverage of the top-26", which is why §3 treats 78% as close to the achievable ceiling for *this* allocation).

So the interesting question is not "improve admission" - that knob has been explored and is at a local optimum - but
**"is 26-per-layer the right allocation?"**.

## 3. Four levers, with their ceilings

| # | lever | ceiling | state |
|---|---|---|---|
| 1 | **more slots** | none: 2.2 GB is the ceiling - 2.3 GB was measured **2/2 failing** (warmup leaves 1 744 MiB free against a 2 068 MiB plan, so prefill goes to the CPU and the round produces 0 tokens) | closed (M20) |
| 2 | **admission / promotion** | small: `cap 8` cost 720 MB of d2d per step and killed prefill throughput; `cap 1 + min_hits 4` beat both ends; the second-sighting rule exists because eviction hurt | tuned, local optimum |
| 3 | **allocation across layers** (shared pool instead of 26 fixed) | **unknown, plausibly large**: hot layers could take 40 slots and cold ones 10 at zero cost, since the total stays 1 248. Every layer's routing skew is different and nothing forces them to be equal | **the one unmeasured lever** |
| 4 | **bytes per miss** | 1.88 MB per miss today (gate + up + down). Splitting the miss so only `gate`+`up` (1.25 MB) must land before the compute starts does not reduce the total, and M26 showed reordering buys nothing while the link is saturated | closed (M26) |

## 4. Experiments, in the order that costs least

**Exp 1 (decisive, one instrumented run).** Add `NAIWA_EXPERT_HIST=1`: a `[48][256]` demand counter on every routing
decision, dumped on shutdown. From that table compute, for the same 1 248 slots:

- `per_layer`: sum over layers of the top-26 coverage -> **the ceiling of today's fixed allocation**;
- `global`: coverage of the global top-1 248 (layer, expert) pairs -> **the ceiling of a shared pool**;
- the per-layer Gini / top-10 share, to see which layers are hot.

Gate: if `global - per_layer` is **under 3 points** of coverage, a shared pool is not worth its complexity and lever 3
closes; the milestone then reports that the numerator cannot be reduced without more memory and recommends stopping the
line here rather than inventing work.

**Exp 2 (cheap A/B, no new mechanism).** Same total slots, allocation skewed by the profile: 40 slots for the hottest
8 layers, 20 for the rest. Purely a re-allocation - no policy change, no extra bytes - so if Exp 1's gap is real this
should show it in `distinct`/misses immediately.

**Exp 3 (flags only).** Re-sweep `NAIWA_PROMOTE_MAX` in {0, 1, 2} x `NAIWA_PROMOTE_MIN_HITS` in {2, 4, 8} with the
measured metric being **MB/step** (not TPOT, which is noisy at +-8%), to confirm §2's local optimum under the current
allocation.

## 5. The honest expectation

Lever 3 is the only one left with real headroom, and its ceiling is *bounded by the skew itself*: if the 48 layers have
similar top-26 coverage, a shared pool gains nothing. The number that decides it is Exp 1's `global - per_layer`, and
it can be computed from a single run with no model change - which is exactly what a milestone that has already produced
three negative results should buy before anyone writes cache code.
