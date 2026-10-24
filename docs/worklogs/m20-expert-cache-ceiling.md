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
