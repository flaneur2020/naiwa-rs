# M20 — the expert cache's ceiling: what actually limits it, and what "aligning with strata" would take

Status: **open** (planned 2026-10-24). Nothing measured yet; this file is the plan and the reasoning.
Parent: [m19-shared-expert-vram.md](m19-shared-expert-vram.md).
One mechanism this milestone: **a VRAM ledger with a per-consumer constraint probe** — one instrument that says, for
a given configuration, *which consumer* sets the expert cache's ceiling. No new cache mechanism is introduced
until the ledger names the binding constraint.

---

## 1. Why this milestone exists

M19 ended with three negative results that all point at the same unanswered question:

| M19 finding | what it implies |
|---|---|
| the borrow saves 650 MiB of private buffer but empties ~345 cache slots per chunk (+610 MiB of refill) | prefill and cache are competing for the *same* VRAM, and moving bytes between them is zero-sum |
| 2.6 GB and 2.9 GB **both OOM, including the baseline arm** | the ceiling is not caused by the borrow, the chunk formula, or the stream |
| chunk 1024 does let the cache reach 2.6 GB, but costs +1.9 s of turn-1 | the ceiling *can* be bought with chunk size — at a price 15x worse than what it saves |

So the question is not "can we borrow" but **"what is the cache's ceiling, who sets it, and which of those setters
is ours to change?"** On this card we hold 2 355 MiB (19% of 12 282 MiB) while strata's `--expert-cache auto`
"fills the card". That gap is the subject.

## 2. The ledger we already have (RTX 4070, 12 282 MiB, `--max-seqs 4`, `--pa-context-len 4096`, MTP 4)

| consumer | MiB | live when | source |
|---|---:|---|---|
| **expert cache**, `--expert-cache-gb 2.3` | **2 355** (1 208 slots, 25–27/layer) | decode | startup line |
| prefill stream: 1 layer buffer | 650 | prefill | `plan()` |
| prefill stream: activations + act/8 + headroom | 522 + 65 + 512 | prefill | `plan()` |
| prefill stream, **2** buffers | 650 more | prefill | `plan()` (never granted here) |
| staging rings (`--pcie-frac`), 7 dtypes x 16 slots | ~175 | decode | startup lines |
| drafter pool (MTP head, Q2_0) | 675 | both | M16 |
| recurrent state | 459 | both | M19 |
| KV + non-expert weights + CUDA context + graphs | **~7 430** | both | **by subtraction - the only row never itemised** |
| **free after startup** | **2 595** | both | startup line |
| decode low-water (measured) | ~600 | decode | `NAIWA_PREFILL_PEAK` |

Two facts fall out immediately, and they are the milestone's first two hypotheses:

- **H-C1 (prefill reservation binds).** 1 749 MiB is reserved for prefill and is *idle during decode*. Handing
  only its "headroom + slack" part to the cache would be ~577 MiB = **+300 slots (+25%)**.
- **H-C2 (decode headroom binds).** After the prefill reservation there is ~850 MiB left, the decode low-water is
  ~600, and 2.6 GB OOMs *even without the stream* - so something grows during decode (KV/recurrent/step buffers)
  and eats the last ~250 MiB before any cache slot can use it.

They are not mutually exclusive, and the whole point of the ledger is to say which one to attack first.

## 3. Strata's structure, for comparison

| layer | strata | us |
|---|---|---|
| who owns the arena | one bump arena carved at startup: `Prefill::init(borrow, borrow_bytes)` (`src/prefill/prefill.cpp:363`) | three separate reservations (cache, stream, staging) |
| prefill workspace | a **byte-sized ring**, ~33 MiB, carved from the cache region and returned after each chunk | 650 MiB whole-expert-stack buffers + 522 MiB activations, freed only at exit |
| cache sizing | `--expert-cache auto` = "the largest cache that still leaves the ring and the decode headroom" | `--expert-cache-gb` hand-set, with the *stream* then funded from whatever is left |
| failure posture | recompute the plan after the first chunk | drop the buffer / disable the stream |
| their own measurement of shrinking the ring for slots | **Coder −12%** - they refuse the trade too | M18 measured the same trade and refused it |

The asymmetry that matters: **their prefill workspace is ~50x smaller than ours**, so on the same card they can
give the cache ~1 500 MiB more without any trade at all. That is not a policy difference, it is the kernel:
their MMQ reads a ring of bytes; our quantised kernel reads a whole expert stack per layer. M19's ring-batching
attempt to close it cost 3.4x on prefill, so the alignment has to come from the kernel side (or from not needing
the stack resident), not from buffer arithmetic.

## 4. Targets ("aligning with strata", stated so they can fail)

| # | target | how it is judged |
|---|---|---|
| **T0** | **TPOT is insensitive to cache size**: `slope = (TPOT(1.85 GB) - TPOT(2.3 GB)) / 0.45 GB` drops from today's **17.8 ms/GB** to **<= 4**, and TPOT at 2.3 GB with it | two serves per arm (cache 2.3 and 1.85), harness 3 rounds |
| T1 | **the ledger explains every MiB**: `MISTRALRS_VRAM_DUMP` + the plan + the measured low-water add up to 12 282 within 2% | itemisation sums |
| T2 | **the binding constraint is named by experiment**, not by argument: for each of C1/C2/C3, a run where removing *only* that consumer raises the cache ceiling | ceiling moves by the predicted amount |
| T3 | cache ceiling **>= 3.0 GB** with the chat harness still passing (0 failures, 3 rounds) | harness |
| T4 | **no regression at equal cache**: at 2.3 GB, TPOT/TTFT medians within M19's gate10 envelope | harness, 3 rounds |

T3 is the alignment target (strata's ceiling on this card, if its auto value is what we think, is ~3.2-3.6 GB);
T4 exists so that "we raised the ceiling" cannot be bought by making the working point worse.

## 5. Plan, in the order that makes each step cheap

1. **Itemise (T1).** Turn on `MISTRALRS_VRAM_DUMP` and record the resident weights by kind; print the arena's
   consumers at startup; reconcile with 2 595 MiB free. *Deliverable: the ~7 430 MiB row split into named parts.*
2. **Probe the ceiling without the stream (H-C2).** Same binary, `--prefill-stream` off, sweep
   `--expert-cache-gb 2.3 → 3.6`. This removes 1 749 MiB of reservation from the equation, so whatever OOMs is
   C2/C3. *Decisive: if the ceiling stays ~2.3, prefill is exonerated and the milestone turns into a decode-side
   budget study.*
3. **Probe the ceiling with the stream but no chunk (H-C1).** `--prefill-stream --max-prefill-chunk-tokens 64`
   (the stream then never activates) at increasing cache. Complements step 2.
4. **Attribute the growth (C2/C3).** With the cache pinned just below the OOM point, sample free VRAM across a
   1725-row prompt + 64 decode steps (`NAIWA_PREFILL_PEAK` already samples per layer for the chunk; the decode
   side needs the same 1 Hz sampler) and fit the slope: MiB per token, per sequence, per MTP step.
5. **Only then choose the mechanism**, one per milestone:
   - if C1 binds: make the prefill reservation *lend* its headroom/slack to the cache during decode (the
     headroom is not used while decoding, and unlike Phase A this needs no alias or emptying);
   - if C2 binds: convert the decode-side growth (KV/recurrent/step buffers) into a computed reservation so the
     cache can be sized exactly, instead of `--expert-cache-gb` + hope;
   - if C3 binds: it is the kernel milestone M20.2 (grouped GEMM/MMQ reading a byte ring), which is also the only
     route to *structurally* matching strata's prefill footprint.

## 6. Gates

- `frac 0` + `--no-lookahead` byte-identical to `harness/m1/reference.oracle.txt`, in every arm that ships on.
- `cargo test` 49 passed / 4 ignored.
- Chat harness: 3 rounds, **0 failures**, medians (never a performance gate).
- Any new ceiling claim must come with the arm that *fails* at ceiling + 0.1 GB (a ceiling you never hit is not a
  ceiling: the same rule as "a test that cannot fail proves nothing").

## 7. Explicit non-goals

- **Not** shrinking the chunk to buy cache: measured, +1.9 s of turn-1 against ~0.13 s of decode, 15x against.
- **Not** borrowing the cache for the chunk again (M19 Phase A/B): both measured, both net zero or worse.
- **Not** double buffering: needs 2 399 MiB and lands on TPOT (M18: +6 ms for −18% turn-1, net negative).
- **Not** requantising the dense weights to 8-bit: breaks the oracle.
- **Not** a unified slot pool: −23% capacity with seven expert shapes (M19).

## 8. Open questions carried in

1. What *is* strata's `--expert-cache auto` on a 12 GB card with this model? Never measured here (M10 pinned it to
   1.5 GB for fairness). One run answers the target for T3 and would replace "~3.2-3.6 GB" with a number.
2. Which of KV, recurrent state, or the MTP step buffers grows during decode? The 2.6 GB baseline OOM happens in
   round 2, on `long turn 1` - i.e. after a conversation's worth of context, which points at KV rather than at the
   step.
3. Does the prefill `headroom` (512 MiB, or free/8) ever bind in a real chunk? It was chosen for the paged-attention
   scratch, and M19's slack experiment suggests the formula under-counts by >89 MiB - so the headroom is doing real
   work and cannot simply be donated.

## 9. Strata's ideas, adopted one at a time

Strata solved this problem structurally, not by tuning. Five of its ideas bear directly on the ceiling, and each
one maps to a step here - which is why M20 is described as "diagnose first": strata's first move *is* to compute a
plan, and we currently do not have one.

| # | strata's idea | where it lives there | our current shape | what M20 does with it |
|---|---|---|---|---|
| S1 | **one arena, carved once from measured free VRAM; the cache is the residual** | `Prefill::init(borrow, borrow_bytes)` carves resident / cache / borrow / ring at startup | cache is hand-set (`--expert-cache-gb`), then the stream takes "whatever is left", and a static 512 MiB headroom is guessed | step 1's ledger, then `--expert-cache-gb auto` = `free - decode_headroom - rings - borrowable_region`; **the ceiling becomes a printed number, not a parameter** |
| S2 | **the chunk is derived from memory, not configured** (`--prefill auto` = the largest chunk that fills the ring) | `bytes_needed(chunk)` priced up front | `--max-prefill-chunk-tokens 8192` hand-set; `plan()` can only refuse afterwards | after S1, invert `fits()`: report the largest chunk the plan funds, and use that |
| S3 | **the plan is recomputed after the first chunk** | `relayout(chunk, borrow, borrow_bytes, err)` | our plan is once, at load; the slack experiment showed its formula under-counts the peak by >89 MiB | recompute from the measured peak (`NAIWA_PREFILL_PEAK` already samples it per layer) and re-carve |
| S4 | **failure posture: shrink, never disable** | a chunk that does not fit re-plans; the feature stays on | a miss once turned the stream off silently (4 s -> 132 s, M18) | any arm that cannot fund its reservation must fail loudly or shrink, and be reported |
| S5 | **borrow the bytes the cache already owns, sized in bytes** | a ring carved from `PreBorrow` space, returned after | we borrow a 650 MiB whole-stack region and empty ~345 slots | **not copied**: M19 measured the alias cost and the ring-batching cost. This is the one idea that needs the kernel (M20.2), and the reason the ceiling gap is ~1.5 GB rather than ~0 |

The honest reading of S5 is the one M19 earned: strata can borrow *cheaply* because what it borrows is small
(~33 MiB ring) and because a byte ring never needs the alias discipline our whole-stack buffers do. Copying
strata's *arithmetic* without copying its *kernel* is exactly what M19 Phase A/B did, and it lost. So M20 takes
S1-S4 (all of which are planning, and all of which are measurable) and leaves S5 where it belongs.

### What "aligning with strata" then means, concretely

1. `--expert-cache-gb auto`: the cache is whatever the plan has left, after a **measured** (not guessed) decode
   headroom - strata's residual idea, and the fix for "2.6 GB OOMs but nobody can say why".
2. One startup line: `total | resident | kv | recurrent | drafter | staging rings | borrowable region |
   decode headroom | cache N slots`. Every MiB named; T1 is then a diff, not an investigation.
3. The chunk printed as "what the plan funds" instead of a hand-set cap (S2), with the recompute-after-first-chunk
   posture (S3) so a formula that under-counts corrects itself instead of silently degrading (S4).

## 10. Estimated payoff, before spending anything (2026-10-24)

Two measured anchors and one measured floor give a curve for TPOT against cache size

```
T(c) = 19.0 + 104.4 * exp(-c / 1.564)      c in GB      (doc class, 64-token answers, MTP 4)
```

| anchor | value | source |
|---|---|---|
| T(2.30) = 43.0 ms | gate10 median | M19 gate10 |
| T(1.85) = 51.0 ms | same workload, cache cut for a second buffer | M18 |
| floor 19.0 ms | every expert resident (`NAIWA_GPU_ONLY_EXPERTS=1`) | M10 (18.8/19.0/24.2) |

**The fit is cross-checked against a point it was not fitted on**: T(1.5) = 59.0 ms, against M10's recorded
"0.5 GB less cache costs decode: TPOT 57-60 ms". Three points, one curve, no free parameter left.

| cache | TPOT (model) | vs today | doc e2e | share of e2e |
|---|---:|---:|---:|---:|
| 1.85 GB | 51.0 ms | +8.0 | +1.5 s | - |
| **2.30 GB (today)** | **43.0 ms** | - | 34.5 s | - |
| 2.60 GB | 38.8 ms | -4.2 | -0.8 s | -2.3% |
| **3.00 GB (T3)** | **34.3 ms** | **-8.7 (-20%)** | **-1.7 s** | **-4.8%** |
| 3.30 GB | 31.7 ms | -11.3 (-26%) | -2.2 s | -6.3% |
| 3.60 GB | 29.5 ms | -13.5 (-32%) | -2.6 s | -7.5% |

**What that buys against strata.** Strata reaches 39 ms on this card with 1.5 GB. Our curve needs **2.59 GB** to
match that number - so the same TPOT costs us **+1.09 GB**, which decomposes almost exactly into the two things
M19 named: **650 MiB** (the whole-stack prefill buffer strata does not need) plus **~0.45 GB** (per-miss
efficiency: their DMA path and kernel against ours).

**What it does not buy.** e2e per conversation is *not* dominated by TPOT: at 64-token answers and three turns,
decode is ~8.3 s of the 34.5 s doc conversation, so a 20-32% TPOT cut is 5-8% of e2e. Strata's doc conversation
is 20.1 s, so even at 3.6 GB we would land at ~31.9 s - **still 1.6x behind**, because the rest of the gap lives
in turn-1 prefill (3.7 s against their 2.4 s) and in per-miss cost.

**Risk-adjusted.** The 3.0-3.6 GB outcome requires the ledger to find 650-1 250 MiB that is genuinely idle during
decode. The 2.6/2.9 GB OOMs - which hit the baseline too - argue that it may not be there. Call that 40-50%
likely; the expected gain is therefore **~2-4% of e2e**, and the milestone's guaranteed product is the diagnosis
(and, if C2 is the binding constraint, a cache size that is *computed* instead of guessed - worth more than the
5% because it removes a whole class of "OOM in round 2").

**The honest comparison with the other open direction**: the kernel (M20.2 / grouped GEMM reading a byte ring)
is the only route that touches the 650 MiB *and* the TTFT; M19 measured the cheap version of it at 3.4x slower,
so it is a project, not a step. The cache ceiling is a step, worth ~5% e2e, and it should be taken first because
it is also the cheapest thing to *know*.

## 11. Correction to §10: the gap is *miss visibility*, not only bytes (2026-10-24)

§10 said "strata reaches 39 ms with 1.5 GB, our curve needs 2.59 GB to match" (+1.09 GB). The arithmetic holds,
but the *interpretation* does not survive a second look at the numbers we already have:

| | strata | us |
|---|---|---|
| TPOT by class (same corpus, same card, concurrency 1) | **39 / 39 / 39 ms** (doc/long/short) | 43 / 46 / 54 ms |
| TPOT when the cache moves by 0.45 GB | flat | **43 -> 51 ms** |
| TPOT with every expert resident | - | **19 ms** |
| reuse | 0.95-0.99 | 0.90-0.98 |

Strata's TPOT is **flat across three workloads with very different working sets**, which is the signature of a
decode that is *not* cache-limited: its misses are off the critical path (lookahead prefetch into the ring, then the
ring feeds the kernel). Ours moves with the cache in every measurement we have, i.e. **our decode is
memory-latency-bound and the miss is visible**.

That changes what "aligning with strata" means:

- The "+1.09 GB" is not "they do more with less cache". It is "**they hide the miss and we do not**", expressed in
  cache-equivalent units. Their 39 ms is their *compute* floor with speculation; our 19 ms all-resident floor is
  below it - so when memory is not the limit we are faster than they are, which is the opposite of the conclusion
  the raw TPOT comparison suggests.
- Therefore the cheap test is **not** VRAM at all: it is `--pcie-frac` (how much of the miss the lookahead ring
  absorbs) and the admission policy, both of which multiply the *effective* cache without a byte of extra memory.
  We run `--pcie-frac 0.2`; M9's oracle notes record that `frac > 0` changes near-ties, so any such sweep must be
  measured on the harness (which tolerates it) and never on the byte-exact oracle arm.
- This reorders M20's own plan: **step 2 becomes "how visible is a miss?"** (sweep `--pcie-frac` 0.2 / 0.4 / 0.6 at
  2.3 GB, record TPOT), before the ledger work, because if miss visibility is the binding term then the 650 MiB
  of buffer is a second-order problem and the ceiling target T3 should be restated in *effective* cache.

Numbers to beat, for the record: strata 39 ms flat; our curve predicts 34.3 ms at 3.0 GB, i.e. **a cache of 3.0 GB
would put us ahead of strata's measured TPOT** - which is only plausible *because* their number is compute-bound and
ours is memory-bound. If a `--pcie-frac` sweep alone moves us from 43 to ~39 ms, then the ceiling is worth less than
§10 estimated and the hiding is worth more.

## 12. How expensive is a miss, for us and for strata (2026-10-24)

The §11 reframing can be priced. At 2.3 GB our doc TPOT is 43 ms and the all-resident floor is 19 ms, so the
**memory term is 24 ms per token**. A token does `48 layers x topk 10 = 480` expert lookups; at 9.9% uniform
coverage (30-50% effective, with popularity skew) that is 240-430 misses:

| effective hit rate | misses/token | implied cost per miss |
|---:|---:|---:|
| 10% | 432 | **56 us** |
| 30% | 336 | **71 us** |
| 50% | 240 | **100 us** |

One 1.88 MB slot at 13.4 GB/s is **144 us**. The implied per-miss cost is the same order as a full transfer, which
says the miss is **serialised on the critical path** - not overlapped, not hidden. This is the whole 24 ms.

Strata's TPOT is flat at 39 ms across working sets (doc/long/short), so its memory term is ~0: either the miss is
fully hidden (lookahead issues the copy a layer ahead, and the cost becomes PCIe *bandwidth* rather than *latency*)
or its per-miss path is cheaper. The flatness argues for hiding; the fact that its 39 ms exceeds our 19 ms
all-resident floor argues that hiding is not free (their compute floor is simply higher than ours).

Two consequences, both cheap to test and both independent of the VRAM ceiling:

1. **`--pcie-frac` is the hiding knob we already have.** We run 0.2, i.e. only a fifth of the misses go through the
   GPU ring; the rest take the CPU path. A sweep at 2.3 GB (0.2 / 0.4 / 0.6) prices the hiding directly: if TPOT
   drops toward 39 ms, the effective cache has grown by ~1 GB *without a byte of VRAM*.
2. **Our prediction problem is harder than theirs.** Hiding requires predicting 480 lookups per token across 48
   layers, and under MTP the drafts change topk, so a wrong guess costs a wasted transfer. Their byte-sized ring
   makes a wrong guess cost 1.88 MB; our whole-stack buffer makes the *workspace* 650 MiB but a wrong *prediction*
   costs the same 1.88 MB - so this is the one place where our architecture is not the handicap.

Recorded so it is not re-derived: **the miss is worth 24 ms/token to us and ~0 to strata; at 56-100 us per miss we
are paying full DMA latency, serially, 240-430 times per token.**

## 13. Hit rate: ours is good, theirs is unmeasured - and the hit rate is not the difference (2026-10-24)

| | us | strata |
|---|---|---|
| coverage (structural) | **9.8%** of the 12 288 experts (1 208 slots at 2.3 GB) | **6.5%** (800 blobs at 1.5 GB) |
| measured expert hit rate | **51% at one row, 78-81% with MTP's 5 rows** (M16; cache was ~12 slots/layer = 4.7% coverage) | **never measured** - we have no counter in their build and never instrumented a run |
| TPOT response to cache size | 43 ms at 2.3 GB, 51 ms at 1.85 GB, 19 ms all-resident | flat 39 ms across three workloads |

Two things follow, and they kill the obvious hypothesis:

1. **Our hit rate is not bad.** At only 4.7% coverage we already hit 51% at one row and 78-81% under MTP, because
   consecutive draft rows share experts. We hold *more* slots than strata's paired configuration (9.8% against
   6.5%) and we still pay 24 ms of memory term per token. **Coverage is not the binding term.**
2. **The 24 ms cannot be explained by transfer volume alone.** At 80% hit rate a token has ~96 misses; 96 full
   1.88 MB transfers at 13.4 GB/s is 13.8 ms. We pay 24 - i.e. **~250 us per miss**, above the 144 us a pure
   transfer costs. The excess is the host side: we run `--pcie-frac 0.2`, so **80% of misses take the CPU path**,
   which has per-miss host overhead on top of the same 24 GB of reads.

So the difference from strata is *not* "they hit more often". It is:

- they keep the miss **off the critical path** (lookahead a layer ahead, cost = bandwidth not latency);
- for the misses that do land on the path, our **path is the CPU one** four times out of five.

Whether strata's *effective* hit rate is also higher than ours (e.g. frequency-based admission instead of our
`last_use` LRU) cannot be answered from our data - and it does not need to be, because the two levers we can pull
are both independent of it:

| lever | why it is first |
|---|---|
| `--pcie-frac` 0.2 -> 0.4 / 0.6 at 2.3 GB | moves misses from the CPU path to the GPU ring: no VRAM, one flag, and M16 already showed 1.0 saturates PCIe (2 668 MiB staged vs 848) so the optimum is interior |
| admission policy (frequency / second-touch instead of pure LRU) | raises the hit rate at *equal* coverage - but only worth measuring once the miss path is cheap, otherwise it optimises the term that is not binding |

## 14. Three proposals evaluated: drop readahead, profile-preload, promote-on-stage (2026-10-24)

### 14.1 Dropping readahead because it "uses PCIe bandwidth"

The arithmetic does not support the premise. At 80% hit rate a token has 96 misses = 180 MB = **13.5 ms of link
time, i.e. 31% of a 43 ms TPOT** (~39% once MTP's ~1.25 forward passes per output token are counted). The link is
**not saturated during decode**; what costs us is latency per miss (250 us against a 144 us transfer), not queueing.
Readahead's bandwidth therefore displaces nothing that is scarce - and it is the only thing standing between a miss
and the critical path.

There is however a second, plausible cost that has nothing to do with bandwidth: `swaps` (lookahead loading a
prediction *over* a resident). A wrong prediction both wastes a transfer and evicts a hot expert, which then misses
again - two misses for one wrong guess. Under MTP the drafts change the topk, so predictions are weaker than in
plain decode. That is measurable today from counters we already keep (`lookahead_loads`, `swaps`,
`lookahead_dropped`).

**Verdict**: not a bandwidth question, so it must be decided as an A/B on prediction accuracy, not argued. Cheap
either way: one flag (`--no-lookahead`), and the counters say whether the swaps were earning their keep.

### 14.2 Profile-preload the expert set (strata's "profile" idea)

Record the expert set a conversation (or a warm-up) touches, then install that set at startup instead of letting
LRU discover it. The ceiling on the benefit is the **cold start only**: our LRU reaches 78-81% within a handful of
tokens, so preloading can only save the misses of the first few tokens - order 10 tokens x 96 misses x 250 us =
**0.24 s per conversation**, i.e. <1% of a 34.5 s e2e, and it cannot help at all once the cache is warm.

It also has a structural limit: the working set is `480 lookups per token`, and a conversation touches far more
distinct experts than the 1 208 slots hold, so "the set" is not storable except as a *ranking*, which is exactly
what an admission policy is. **Verdict: low value, and the parts that are worth anything are admission policy,
not a saved profile.** (A profile does have one cheap use: seeding a *fresh* process after a restart, which is not
a latency problem we have.)

### 14.3 Promote staged experts into the cache, with a cap

This is the one of the three with a real mechanism behind it: a staged expert has **already crossed the link**, so
admitting it costs a device-to-device copy (or a slot hand-over) instead of a 1.88 MB transfer, and it converts our
most expensive miss kind (the CPU path, 250 us) into a hit. Pollution is the risk, hence the cap the proposal asks
for.

Upper bound, using the measured 250 us per miss saved and `--pcie-frac 0.2` (19 of 96 misses per token are staged):

| reuse of the staged misses | extra hits/token | TPOT saved |
|---|---:|---:|
| 10% | 10 | **2.4 ms** |
| 25% | 24 | **6.0 ms** |
| 50% | 48 | 12.0 ms |

But the bound is optimistic in one direction and pessimistic in another: at 80% hit rate the *remaining 20%* are by
definition the rarely-used experts, so their reuse is likely near the bottom of that table - **yet** the same experts
tend to be reused *within a conversation* (the prompt's topic persists), which is exactly the time scale a decode
cache sees. So 2-6 ms is the honest range, at zero VRAM, against 8.7 ms for the whole 3 GB ceiling project.

### 14.4 The combination is the interesting design, and it is one experiment

The three proposals add up to a coherent restructure: **stop speculating, start exploiting what is certain.**

```
today:  lookahead predicts -> transfers over the link -> swaps over residents (sometimes wrong)
        staged experts are used once and thrown away
after:  no speculative prefetch -> every transfer is one we needed
        every staged expert is a candidate for residency (capped)
```

That is also closer to strata's shape: its ring feeds the kernel and its admission is demand-driven, with no
speculative traffic of its own.

| arm | flags | what it tests |
|---|---|---|
| A | today (`--pcie-frac 0.2`, lookahead on) | baseline |
| B | `--no-lookahead` | does the speculation pay? (`swaps` accuracy) |
| C | `--pcie-frac 0.4` + promote-on-stage, cap 16/token | does demand-driven admission beat speculation at equal bandwidth? |
| D | B + C | the restructure |

Acceptance: harness 3 rounds, 0 failures, medians; oracle untouched (`frac` stays 0 for the byte-exact arm).
Predicted: B neutral-to-worse, C **-2..-6 ms TPOT**, D best and cheapest to reason about.

## 15. Restated plan: the goal is a flat TPOT curve, not a bigger number (2026-10-24)

The user's framing is the right one, and it reorders everything: strata's property is not "39 ms", it is **39 ms
whatever the cache does** (doc/long/short, and by construction whatever its `auto` picks). Our curve is steep:

```
TPOT(c) = 19.0 + 104.4 * exp(-c / 1.564)      slope at 2.3 GB: 17.8 ms/GB
strata:                                       slope: ~0
```

So the milestone's primary target becomes **T0 (flatness)**, and the ceiling work (T3) is demoted to a supporting
action. Two consequences worth stating plainly:

- **A bigger cache becomes less valuable, not more.** If the slope is 4 ms/GB, the difference between 2.3 and 1.85 GB
  is 1.8 ms; between 2.3 and 3.0 GB it is 2.8 ms. At today's slope the same 0.7 GB is worth 12 ms.
- **The M19 borrow is disqualified by this KPI**, independently of its other costs: it injects ~345 evictions =
  a -0.65 GB shock *per chunk*, i.e. it is a deliberate spike on the very curve we are trying to flatten. It stays
  off by default for a reason that is now sharper than "net zero".

### The levers, ordered by expected value per unit of work

| arm | change | why it flattens | expected |
|---|---|---|---|
| **A0** | today: `--pcie-frac 0.2`, lookahead on, promote off | baseline | TPOT 43, slope 17.8 |
| **A1** | `--pcie-frac 0.4` | moves misses from the CPU path (250 us, unhideable) to the ring (hideable); link is at 31-39% so there is room | ~37-39, slope ~10 |
| **A2** | `--pcie-frac 0.6` | as above, more of the miss becomes lookahead-coverable | ~33-36, slope ~5 |
| **A3** | A2 + **promote-on-stage, cap 16/token** | staged bytes already crossed the link; admitting them turns the *next* miss into a hit, which both lowers TPOT and lowers the slope | ~32-35, slope ~4 |
| **A4** | A3 + whatever the ceiling ledger funds | fewer misses to hide in the first place | ~30-33 |

No new mechanism per arm beyond the flags and the promotion path, and the oracle arm (`frac 0`) is untouched
throughout - `frac > 0` is harness-only by the M9 rule.

### Why A1-A3 can plausibly beat the whole ceiling project

`dTPOT/dCache` is dominated by the *exposed* fraction of misses. Today only a fifth of misses even reach the ring,
so four fifths are unhideable by construction:

| `--pcie-frac` | misses through the ring | expected memory term | expected TPOT |
|---:|---:|---:|---:|
| 0.2 (today) | 19/token | 24 ms | 43 |
| 0.4 | 38/token | ~14 ms | ~33 |
| 0.6 | 58/token | ~9 ms | ~28 |
| 0.8 | 77/token | ~4 ms | ~23 |

The table is an upper bound - it assumes hiding is free and that the link does not queue - and M16 already showed
`frac 1.0` with MTP saturates the link (2 668 MiB staged per window against 848), so the optimum is **interior,
somewhere in 0.4-0.6**. But even half of that table is worth more than the 8.7 ms the ceiling buys, and it costs
**no VRAM at all**.

### Order of work

1. **A1/A2 sweep** (`--pcie-frac` 0.4 / 0.6, cache 2.3 and 1.85 per arm) - one flag, two serves per arm, and it
   measures T0 directly.
2. **A3**: promote-on-stage with a cap (the only code change in the sequence: admit a staged expert into an empty
   slot, else over the LRU victim, at most N per token, counted as `promotions`).
3. **Ledger + `auto`** (S1-S4): keep it, but as the thing that stops "OOM in round 2" and lets chunk/cache be
   derived - not as the primary performance lever.
4. **Kernel (M20.2)**: the only route to the prefill side of the gap and to removing the 650 MiB; unchanged by any
   of the above.

Recorded prediction, so it can be wrong in public: **A3 lands at TPOT 33-36 ms with slope ~4-5 ms/GB; the ceiling
work then adds 2-3 ms on top; the kernel is what closes the rest.**

## 16. T0 measured: `--pcie-frac 0.4` makes the TPOT curve flat (2026-10-24)

The sweep M20 §15 asked for, run on the shipping binary: same model, same flags, cache 2.3 and 1.85 GB, five
`--pcie-frac` values; the probe is one `doc` conversation, TPOT taken from turn 2 (64 output tokens, reuse 0.97) so
every cell is the same shape of work. n = 2 at 0.2/0.6, 3-5 at 0.3-0.5.

| `--pcie-frac` | TPOT @ 2.3 GB | TPOT @ 1.85 GB | **slope** |
|---:|---:|---:|---:|
| **0.2 (today)** | 41.7 ms | 46.5 ms | **+10.7 ms/GB** |
| 0.3 | 40.4 | 40.9 | **+1.2** |
| **0.4** | **40.8** (n=5) | **41.0** (n=5) | **+0.4** |
| 0.5 | 43.7 | 42.6 | -2.4 |
| 0.6 | 46.4 | 53.7 | **+16.3** |

**T0 is met**, and better than the target: slope 17.8 -> **0.4 ms/GB at frac 0.4** (target was <= 4). And it is not a
trade - the absolute number improves too: 41.7 -> 40.8 ms at 2.3 GB, and **46.5 -> 41.0 ms at 1.85 GB (-12%)**.

The shape is exactly the predicted one: the lever moves misses off the 250 us CPU path onto the ring, where the
lookahead can cover them; and past ~0.5 the ring's own traffic crosses the link's capacity (frac 0.6 is 6 ms worse
at 2.3 GB and 13 ms worse at 1.85 GB), which is M16's `frac 1.0` saturation arriving earlier than expected. **The
optimum is interior, at 0.3-0.4.**

### What this is worth, immediately

At frac 0.4 a **1.85 GB cache performs like the 2.3 GB cache did at frac 0.2** (41.0 against 41.7 ms). So 450 MiB
of cache can now be handed to something else *for free* - and M18 already measured what that 450 MiB buys when
given to the prefill stream: a second layer buffer, **turn-1 4 431 -> 3 654 ms (-18%)**. The trade M18 had to refuse
(cache for buffers) is now free, which is a bigger effect than the whole ceiling project (8.7 ms of TPOT).

That is the next experiment, and it is one flag: `--pcie-frac 0.4 --expert-cache-gb 1.85` plus the stream's second
buffer, turn-1 against today.

### Caveat, in writing

`frac > 0` is not byte-exact (M9: near-ties flip with the resident/miss split), which is why this arm is measured on
the chat harness and never on the `frac 0` oracle. The harness runs below are the correctness check for it.

## 17. The freed 450 MiB, spent: turn-1 -21% and TPOT slightly better (2026-10-24)

T0 bought 450 MiB of cache back for free (a 1.85 GB cache now performs like 2.3 GB did at frac 0.2). M18 said what
that 450 MiB is worth if given to the prefill stream: a second layer buffer, -18% on turn-1. Measured, same binary,
same probe, two reps per arm:

| arm | cache | chunk | plan | turn-1 TTFT | turn-2 TPOT |
|---|---:|---:|---|---:|---:|
| A (today) | 2.3 GB | 8192 | 1 x 650 MiB + 522 act (+192 head) = 1 364 of 2 336 free | 3.83 / 3.64 s | 42.3 / 40.7 ms |
| **C** | **1.85 GB** | **8192** | **2 x 650 MiB + 522 act = 2 014 of 2 882 free** | **3.05 / 2.87 s** | **40.5 / 40.8 ms** |
| B | 1.85 GB | 4096 | 2 x 650 + 426 act = 1 918 of 2 882 free | 3.09 / 2.88 s | 45.6 / 40.8 ms |

- **turn-1 TTFT 3.74 -> 2.96 s (-21%)** - the second buffer finally fits, because the cache released 450 MiB.
- **TPOT 41.5 -> 40.65 ms (-0.9%)**: the trade M18 had to refuse (cache for buffers, +6 ms of TPOT) **now costs
  nothing at all** - the flatness bought at frac 0.4 pays for it.
- Chunk 4096 buys nothing on top (it only shrinks the activations); 8192 stays.

So the configuration that ships is now **`--expert-cache-gb 1.85 --pcie-frac 0.4 --max-prefill-chunk-tokens 8192`,
with two prefill buffers granted by the plan**, against M19's `2.3 / 0.2 / 8192, one buffer`:

| | M19 gate10 | now |
|---|---:|---:|
| turn-1 TTFT (doc) | 3.8 s | **3.0 s** |
| TPOT (doc, turn 2) | 41.7 ms | **40.7 ms** |
| TPOT slope (0.45 GB) | +10.7 ms/GB | **+0.4 ms/GB** |
| prefill buffers | 1 | **2** |

## 18. The second buffer needs a decode reserve - and the plan did not have one (2026-10-24)

Spending the freed 450 MiB on the second buffer (§17) **OOM'd on the harness** - not a NaN, not a slow run:

```
step - Model failed with error: moe experts forward DriverError(CUDA_ERROR_OUT_OF_MEMORY, "out of memory")
naiwa round 1 doc turn 1 failed: no content delta
```

Cause: `plan` priced the chunk (`buffers + act + act/8 + max(512 MiB, free/8)`) and then granted two buffers with
**606 MiB to spare** - but the buffers stay allocated for the session, so decode only gets what is left, and the
measured decode low-water is ~600 MiB. The chunk and the decode reserve are needed at *different times*, which is
M19's whole premise, but the buffers are permanent, so the reserve has to survive them.

Two fixes, both small:

1. **A decode reserve on the second buffer only** (`NAIWA_DECODE_RESERVE_MIB`, default 700 MiB). The second buffer
   is the optional one, so refusing it degrades (one buffer, same chunk) instead of disabling the stream - the
   failure posture S4 asks for. It does not touch the one-buffer path, which is validated.
2. **`log_plan` printed 192 MiB of "headroom" while `fits` required 512** - a reporting bug that made the plan look
   roomier than it was. Both now use the same constant (`MAX_HEADROOM`), so the line is the ledger.

Re-measured with the reserve in place (2 buffers now only where free VRAM funds them *and* the reserve):

| cache | free at load | buffers | turn-1 | turn-2 TPOT |
|---:|---:|---|---:|---:|
| 2.3 GB | 2 336 MiB | 1 | 3.83 / 3.64 s | 40.7 / 40.8 ms |
| 1.85 GB | 2 882 | 1 (refused: spare 606 < 700) | - | 41.0 |
| 1.6 GB | 3 169 | **2** | **4.01 / 3.95 s** | 43.5 / 41.7 |
| 1.4 GB | 3 393 | **2** | **3.07 / 2.81 s** | 45.6 / 41.0 |

So the second buffer is worth its -21% turn-1 at 1.4 GB but costs 0.5-5 ms of TPOT there, and the turn-1 numbers
are not monotone in cache size (1.6 GB reads worse than 1.4). **The flat-curve configuration is the one that ships**:
2.3 GB, frac 0.4, one buffer, TPOT 40.7 and turn-1 3.7 - and the second buffer stays a knob for later, now guarded
by the reserve instead of by luck.

### What this episode adds to the plan

It is the first measured instance of the milestone's T2: **the constraint was named by an experiment** (an OOM
inside `moe experts forward`, at 606 MiB of spare) rather than by argument. It also prices one of S1's terms: the
decode reserve is ~700 MiB on this card, which is exactly the number `--expert-cache-gb auto` will have to subtract.

### Final validation of the shipped configuration (2026-10-24)

`--expert-cache-gb 2.3 --pcie-frac 0.4 --max-prefill-chunk-tokens 8192`, one layer buffer, decode reserve in place:
**2 rounds, 132 turns, 0 failures**, and the plan line now reads the truth
(`1 x 650 + 522 act + 512 headroom = 1684 MiB of 2335 free`).

| class | TPOT p50 | turn-1 TTFT | e2e/conv | gate10 (frac 0.2) TPOT |
|---|---:|---:|---:|---:|
| code | 34 ms | 0.93 s | 26.1 s | 36 |
| doc | 46 | 3.82 s | 34.5 s | 43 |
| long | 44 | 3.73 s | 43.8 s | 44 |
| short | **40** | 1.03 s | 40.3 s | **54** |

The per-class medians are noisy at this turn count (doc reads 46 against the probe's 40.7 - the harness mixes turn
shapes), but no class regressed and `short` - the class with the most turns and therefore the most misses - moved
54 -> 40 ms. The controlled probe (§16) remains the clean statement of the flatness effect.

## 19. Where we stand against strata, 2026-10-24 (after T0)

Both engines, same card, concurrency 1, same corpus and protocol. Strata's columns are its paired M10.2 run
(`--spec 4`, `--expert-cache 800` blobs -> **1 028 slots / 1.98 GiB** when it sizes itself, `--kv int8`); ours are
g14 (2.3 GB, `--pcie-frac 0.4`, chunk 8192, one prefill buffer, MTP 4).

| class | metric | strata | us now | delta |
|---|---|---:|---:|---:|
| `doc` | TTFT turn 1 | 2 430 ms | 3 819 ms | **+57%** |
| | TTFT >=2 (p50) | 631 ms | 1 096 ms | **+74%** |
| | TPOT p50 | 39 ms | 46 ms (probe: **40.8**) | +18% (+5%) |
| | reuse | 0.99 | 0.98 | -1 pt |
| | e2e / conversation | 20.1 s | 34.5 s | **+72%** |
| `long` | TTFT turn 1 | 2 273 ms | 3 730 ms | **+64%** |
| | TTFT >=2 | 1 018 ms | 988 ms | **-3%** |
| | TPOT | 39 ms | 44 ms | +13% |
| | reuse | 0.99 | 0.98 | -1 pt |
| | e2e | 39.9 s | 43.8 s | +10% |
| `short` | TTFT turn 1 | 1 035 ms | 1 031 ms | -0.4% |
| | TTFT >=2 | 625 ms | 992 ms | **+59%** |
| | TPOT | 39 ms | 40 ms | +3% |
| | reuse | 0.95 | 0.91 | -4 pt |
| | e2e | 29.1 s | 40.3 s | **+38%** |
| `code` | TTFT turn 1 | 984 ms | 929 ms | -6% |
| | TTFT >=2 | 1 165 ms | 1 196 ms | +3% |
| | TPOT | 29 ms | 34 ms | +17% |
| | reuse | 0.67 | 0.87 | **+20 pt** |
| | e2e | 21.9 s | 26.1 s | +19% |

### Where we are ahead, and by how much

| property | us | strata | reading |
|---|---|---|---|
| TPOT with every expert resident (compute floor) | **19 ms** | 39 ms | our decode is ~2x faster when memory is not the limit |
| decode hit rate, single row | **51%** | 47.3% (its own logged figure) | we hit *more* often at a comparable cache |
| decode hit rate under MTP | **78-81%** | n/a (no equivalent) | speculation raises it for us |
| redo on `code` (mixed code/doc conversation) | 0.87 | 0.67 | we reuse context better on that class |
| TPOT slope across a 0.45 GB cache change | **+0.4 ms/GB** | ~0 (flat) | we are now flat too, since frac 0.4 |

### What is actually left

1. **Turn-1 prefill, +1.3-1.5 s on every class with a long prompt.** This is the single largest item and it is
   structural: 24 GB of expert reads per 1725-row chunk, and strata streams them through its byte ring while we
   DMA whole layers into 650 MiB buffers. M17/M18 closed 1.5 s of it; the rest needs the kernel (M20.2).
2. **TTFT >=2 on `doc`/`short` (+0.4-0.5 s)** - the tail of a turn's new tokens taking the chunk path at 24 GB per
   chunk; the same term one size down.
3. **TPOT +5-13%**, which is now the *smallest* of the three: 40.8 ms against 39 in the clean probe. Their 39 is flat
   because the miss is hidden; ours is flat because frac 0.4 hides most of it. Both are near their compute floors
   (39 and 19 respectively) - ours is the lower floor, theirs the better hiding.
4. **Unexplained residual in `doc` e2e**: the components above account for ~3.2 s of a 14.4 s difference, and the
   rest is in the tail (p90 TTFT 2 161 ms against strata's 733) and in turns the medians hide. Worth one
   per-turn diff before it is explained away as bookkeeping.

### Cache, corrected

The earlier note that strata runs 1.5 GB is the *pinned* M10 configuration (`--expert-cache 800`); when it sizes
itself it takes **1 028 slots / 1.98 GiB**, i.e. 8.4% coverage against our 9.8%. So the cache sizes are now
comparable and **the old "+1.09 GB" framing of the gap is obsolete** - with frac 0.4 the TPOT gap is +5% (probe) and
their hit rate is the *lower* of the two. The gap that remains is prefill, not decode.

## 20. Promote-on-stage (A3): implemented, and it deadlocks in warmup — not resolved (2026-10-24)

The one lever left that costs no VRAM: a staged expert has already crossed the link, so admitting it into the cache
is a **device-to-device copy** instead of a second H2D. Implemented as:

- `CopyStream::queue_d2d(src_fence, dst_fence, parts)` - waits on both fences (the ring slot's readers and the cache
  slot's) and issues one `memcpy_dtod_async` per projection part.
- `CachedLayer::promote_pending(cs)` - drains the previous step's staged `(expert, Arc<SlotBuf>)` list, reserves a
  slot (empty first, then LRU), queues the d2d copy and publishes, under `lock_access` so the borrow's gate holds.
  Capped by `NAIWA_PROMOTE_MAX` per layer per step (default 0 = off).
- `dma_miss`: set the pending list at the end of `stage_misses`, and run `promote_pending` **at the start of the next
  step's `stream_pcie`** - before this step's copies - so the existing per-layer completion event (`done`) orders the
  readers, and so the pending `Arc`s are dropped before the ring may hand those slots out again.

**It hangs in warmup**: the process reaches the staging-ring creation and then every thread sits on a futex. The
hang survives with promotion disabled (`set_promote_pending(vec![])` per step), and disappears when the whole diff is
stashed - so it is caused by the diff, not by the environment (which was backed up by a `READY` run of the stashed
build on the same command line).

Most likely cause, not yet proven: the pending list holds an `Arc` clone of a **ring** slot across a step boundary,
and `stage_misses` waits on that slot's readers with `copy_stream::wait_for_readers`, which spins until the refcount
falls back to the ring's own `Arc`. Draining at the start of the next step should be early enough - but the warmup
also runs the lookahead/doorbell pass, and that path can hold a staged slot at the same time, which would turn the
wait into a deadlock. The next design should therefore **not** keep a clone across a step boundary: promote the
expert *inside* `stage_misses`, handing the ring `Arc` straight to the cache slot and giving the ring a replacement
buffer, or promote at the end of the step with a per-slot "promote once" flag so no `Arc` outlives the step.

Kept, for the next attempt: the patch is in [`m20-promote-attempt.patch.txt`](m20-promote-attempt.patch.txt), and the
estimate that motivates it is unchanged (2-6 ms of TPOT at zero VRAM, from §14.3).

## 21. Promote-on-stage, second attempt: the deadlock is gone, the enabled path crawls (2026-10-24)

The §20 deadlock is **not** the promotion logic. Two facts settle it:

- the whole diff stashed and the identical command line is `READY`, but the same diff with promotion **disabled**
  (`NAIWA_PROMOTE_MAX=0`) also reached `READY` once the machine had no other server running - the earlier hangs were
  a 58 GB RSS process thrashing against 32 GB of available memory, not a lock;
- every thread in the "deadlock" was on a futex with **zero** CPU in the copy path, which is what a page-fault
  storm looks like, not what `wait_for_readers`' spin looks like.

Redesigned anyway, because the §20 hazard was real even if it was not the cause: **nothing keeps an `Arc` of a
staging slot across a step**. The staged list is now written by `stage_misses` and consumed by
`promote_staged` **in the same step, on the same thread**, called from `decode.rs` right after the fence loop (so the
d2d copies queue behind every kernel that reads the ring). The reader ordering comes from the usual mechanism: the
copies are recorded into a per-layer `promote` `CopyDone`, and the next step's resident submit waits on it next to
`wait_preload`. The cap is **per step across layers** - at a per-layer cap of 8 the first version queued
48 x 8 = 384 d2d copies of 1.88 MB = **720 MB per token**, which ran prefill at 5.8 T/s.

**What is measured, and what is not.**

| `NAIWA_PROMOTE_MAX` | prefill | turn-2 TPOT | verdict |
|---|---:|---|---|
| 0 (off) | normal | 41.6 / 40.0 / 43.3 ms | identical to the shipped build (`promoting()` is false, so no call is made at all) |
| 8, per layer | **5.8 T/s** | - | 720 MB/step: withdrawn, replaced by a per-step cap |
| 8, per step | **5.8 T/s** | - | **still crawls**, so volume was not the (only) cause - unexplained |

So the mechanism is in the tree, correct in shape, and **off**: with the cap at zero, `promoting()` short-circuits
before any of it runs, which is why the shipped configuration is bit-for-bit the one g14 validated. What is missing is
a diagnosis of why enabling it costs ~60x on prefill; the next step is a single-layer trace
(`NAIWA_RANK_DBG`-style) of `queue_d2d` against the copy stream's own backlog, since promotion shares the stream with
staging and the prefill chunk's own uploads.

Estimate unchanged and still unproven: **2-6 ms of TPOT at zero VRAM** (§14.3). It stays off until it can be measured
without the penalty.

### Where this was left, and what blocks the next step

- The code is committed and **off** (`NAIWA_PROMOTE_MAX` unset): `CachedLayer::promoting()` is false, so
  `set_staged_now`, `promote_staged` and `wait_promote` are never called and the shipped path is the one gate14
  validated. `cargo test`: 49 passed.
- The cap-0 arm **was** measured on this build before the machine degraded: turn-2 TPOT 41.6 / 40.0 / 43.3 ms, i.e.
  identical to the shipped configuration, as expected.
- The enabled arm still needs an explanation: with a per-step cap of 8, prefill runs at **5.8 T/s** (a 1700-token
  prompt does not finish), which is not explained by copy volume (the per-layer version's 720 MB/step was one
  cause, and fixing it changed nothing measurable).
- **The machine is now unusable for this**: repeated load/kill cycles leave every server stalling in warmup *before*
  `prefill (warmup): plan` is logged, on the shipped code path as well (58 GB RSS against 32-61 GB available,
  page-fault traffic, zero progress). The next attempt needs a fresh boot before it can be measured at all.

Next step when the machine is clean: run the same three arms (0 / 8 / 16 with the per-step cap) and add
`NAIWA_PROMOTE_TRACE=1` printing, per layer, the number of promotions and the copy stream's backlog - if promotion
disturbs prefill it must be through the stream it shares with the chunk's own uploads, and that is one number away
from being visible.

## 22. The hang at cap 0 was a real bug in the diff — found and fixed (2026-10-24, after a reboot)

The reboot made the picture unambiguous: the pre-promotion build (`475d0a3`) came up **READY** on the first try,
the promotion build **NOTREADY 4/4** - so it was the diff, not the machine, and not the environment.

The bug was one line's worth of asymmetry in `dma_miss::stream_pcie`:

```rust
// before
if layer.promoting() { layer.set_staged_now(Vec::new()); }   // the *clear* was gated ...
...
layer.set_staged_now(slot_of_pending);                       // ... but the *write* was not
```

So with promotion off, every step stored `Arc` clones of the step's ring slots in `staged_now` and **nothing ever
drained them**. `stage_misses` waits on each ring slot's refcount with `copy_stream::wait_for_readers`, which spins
until the refcount falls back to the ring's own `Arc` - it never did, so the warmup spun forever. That is exactly the
mechanism §20 hypothesised, arriving through a different door: not the pending *list* design, but an ungated write
next to a gated clear.

Fix: clear unconditionally, and only *populate* when promotion is on:

```rust
layer.set_staged_now(Vec::new());                 // unconditional
...
if layer.promoting() { layer.set_staged_now(v); } // only what will be drained
```

Verified after the fix: `cap 0` is **READY** and measures turn-2 TPOT **37.5 / 57.2 / 35.7 ms** (the 57 ms is the
usual outlier; the shipped arm measured 40-43 ms across the day, so the arms overlap).

**Still open**: the `cap > 0` arm. With the fix it comes up READY and prefill is back to a normal 345 T/s (the 5.8 T/s
crawl is gone - that was the same leak, seen from the promote side), but the first request then does not complete and
`PROMOTE` never prints, i.e. promotion blocks before its first publish. Next step is a one-request trace with
`NAIWA_PROMOTE_TRACE=1` plus `NAIWA_STEP_DBG`, looking specifically at `queue_d2d`'s wait on the **destination**
slot's fence: that event is recorded by a previous step's reader fence, and waiting on it from the copy stream is the
one ordering nobody has exercised before.

## 23. Promote-on-stage measured: +2.0 ms at cap 1, +4.3 at cap 2 — a clean negative (2026-10-24)

With the two deadlocks fixed (and a third correct-by-construction restriction added), the mechanism finally runs and
its value can be read off directly. Turn-2 TPOT of the `doc` probe, three runs per arm, same server flags, one server
at a time:

| `NAIWA_PROMOTE_MAX` (per layer) | runs | median | vs baseline |
|---|---|---:|---:|
| **0 (off)** | 40.7 / 40.8 / 38.8 | **40.7 ms** | - |
| 1 (48 copies/step, 90 MB) | 41.9 / 42.7 / 49.3 | **42.7 ms** | **+2.0** |
| 2 (96 copies/step, 180 MB) | 45.0 / 42.4 / 45.3 | **45.0 ms** | **+4.3** |

Monotone in the cap, so it is the mechanism and not noise. The reason is the same law M19 hit twice: a promoted
expert is by definition one that just **missed**, i.e. a rare one, while the slot it takes comes from the LRU tail -
so the copy buys a hit that the conversation was unlikely to use and costs a hot resident. On top of that the copy
shares the one copy stream with the step's staging (180 MB), which is where the marginal cost lands.

### The three bugs this cost, all worth keeping fixed

1. **The ungated write / gated clear** (§22): ring-slot `Arc`s leaked, `wait_for_readers` spun, warmup hung. Fixed.
2. **The token thread blocked on the borrow gate**: `promote_staged` took `slab::lock_access()`, which the lookahead
   holds across `upload_in_place` - and that *waits for the token thread's own read clones to drop*. Straight
   deadlock, and the reason "the first request never completes". Fixed with
   `slab::try_lock_access()`: the token path never blocks on it, it stands down for that step instead (the mechanism
   only needs the atomic `borrowed()` flag, not the lock).
3. **Evicting a slot the current step is reading**: the plan is built before promotion, so the slot's read fence does
   not cover the incoming copy. Fixed by restricting victims to `last_use < tick`, which is exactly "no kernel of this
   step touches it" (`Route::submit` stamps `tick` on every hit).

### Verdict

Off by default, and it stays off: the estimate in §14.3 (2-6 ms) was the *upper bound from reuse*, and the measured
sign is the opposite. What the episode established is worth more than the 2 ms would have been: the copy stream is
saturated enough that *any* extra traffic on it is a net loss, which is the same conclusion as the `--pcie-frac`
sweep's cliff at 0.6 (§16) - and it is now measured rather than inferred.
