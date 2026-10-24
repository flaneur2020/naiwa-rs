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
