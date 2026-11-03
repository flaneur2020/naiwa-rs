# M33 - the prompt path borrows the expert cache (strata's loan)

**Status:** open. S1-S4 built; §12 and §13 are two rounds of the same investigation and they **contradict each other
on purpose**: §12 priced the chunk from a measurement that was silently contaminated, §13 found the contamination and
replaced both the number and the conclusions of §11/§12. The loan stays **opt-in**, for a reason that survived the
correction (the wipe), not for the one §11 gave (whether it lent at all). Numbering note: M29-M32
are withdrawn (`roadmap.md`, "M29-M32 aborted 2026-11-01"), so this milestone takes the next free number.
**Parent:** M23 (the prompt law `1.15 s per chunk + 1.67 ms per token`), M18 (a second buffer buys overlap, 16-19%),
M20 (the cache ceiling), M10.3 (the chat tail: `TTFT>=2` 1193 ms against strata's 651).
**One mechanism:** the expert cache is sized **at full size**, and a prompt **borrows the tail of that same
allocation** for its layer buffers, refilling the slots it displaced when the prompt ends - instead of withholding
2 x 650 MiB *beside* the cache. strata's `--prefill-borrow`, which is their default.

Today the two phases are **already mutually exclusive in memory** (M18/M33 §2: a chunk uses only its layer buffers and
never consults the cache; decode uses only the cache and never the buffers), so this is a bookkeeping change, not a
capacity change: the bytes are the same, the *owner* alternates.

---

## 1. What strata does, in five parts

| # | part | where | the point |
|---|---|---|---|
| 1 | the cache is sized first, **at full size**; the prompt path reserves nothing | `generate.cpp:3202-3209` | "THE PROMPT PATH'S BUFFERS ARE BORROWED FROM THE CACHE, NOT WITHHELD BESIDE IT ... this reserve is zero" |
| 2 | the loan is the **tail** of that one allocation | `generate.cpp:11170-11174` | `first = xcache.slots() - k`; `borrow = device_slot(first)`; `borrow_bytes = bytes() - slot_offsets()[first]` |
| 3 | `k` comes from a **pure function of the chunk**, capped | `plan_lend` `5776-5851`, `5712` | `k = slots_from_bytes(Prefill::bytes_needed(g, ss, c))`, with `k + 128 <= slots` and `k * 100 <= kAutoLendPct * slots`; **90%** when the copies are DMA from pinned RAM, 85% when host copies are the limit |
| 4 | a loan is **ledgered and refilled** | `generate.cpp:11176-11185`, `9570-9600` | the experts in the lent slots go into `lent`, are marked non-resident, and are copied back by queued copies with **one** wait (`STRATA_REFILL_BLOCKING=1` waits on each) |
| 5 | **chunk and ring share one byte budget** | `5796-5844` | `--prefill auto` bisects the largest chunk that still leaves the streamed ring full; a ring slot is worth ~0.53 tok/s and a chunk token ~0.05 |

Two measurements strata recorded about the cost of a loan, both relevant:

- **A loan is not free per request.** On cards that hold (nearly) every expert of their layers a 2K prompt read
  **37% slower** than with own buffers (`generate.cpp:2067-2072`). On this model the cache holds 26 of 256 experts
  per layer (10%), so nearly everything streams anyway and a lent slot costs only the refill of what it displaced.
- **Bigger chunk, more lent, faster** on a 12 GB card with a 32K Q2_0 prompt: 4096 -> 791 tok/s, 6144 -> 878,
  **8192 -> 973 with 69% of the slots lent** (`generate.cpp:5744-5748`).

---

## 2. What we have today

| part | ours | gap |
|---|---|---|
| 1 full-size cache | `--expert-cache-gb`, and `plan_with_reserve` withholds `stride * layer_bytes` **beside** it (`prefill_stream.rs:119-192`) | the borrow arm exists (`fits`' `if borrowing` branch) but is opt-in via `NAIWA_PREFILL_BORROW=1` |
| 2 tail | `Slab` behind `NAIWA_CACHE_SLAB=1` (`provider.rs:83-111`); `borrow_at(offset, ...)` from the slab's **low end** (`slab.rs:123`) | place the loan at the **tail**, and enable the slab by default |
| 3 pure sizing + cap | `plan_with_reserve` decides chunk and stride **once, at warmup** | per request; add the lend cap |
| 4 ledger + refill | none. `borrow_buffers`' own doc: *"The slots the ranges cover come back empty, exactly as after an eviction"* (`prefill_stream.rs:477-484`) | the whole `lent` mechanism |
| 5 shared budget | none; the chunk is bounded by activations only | needs the layer-buffer equivalent (see §6, S6) |
| mutual exclusion | `BORROWED` flag + `ACCESS` lock (`slab.rs:39-62`) + `quiesce` | **have it**; must be extended to the refill path |

Three structural facts that make the port smaller than it looks:

- **Our banks are contiguous.** `alloc_bank_in` bump-allocates the three projections of a layer back to back
  (`staging.rs:137-160`), so "the last N MiB of the slab" *is* the last few layers' whole banks - one byte range,
  which is exactly what `borrow_from` wants. No blob re-layout is needed for part 1-4.
- **We do not need strata's RAM complement.** Their `pin_cache_complement` exists because their slots' experts may
  live in a file/SSD tier; our slots are copies of the **page-locked host stacks** that are always mapped, so a refill
  is one PCIe copy (650 MiB at 13.4 GB/s = 48 ms).
- **The failure modes are known.** The borrow already carries its own pricing (below), including one recorded kill.

### The borrow pricing that is already in the tree

`prefill_stream.rs:150-163`, measured: with a borrowed layer buffer the chunk still pays **`act * 2 + head`** out of
free VRAM because the step's real scratch is ~1.95x the activation estimate (866 MiB for a 1696-row chunk whose
`act` is 444). Its comment records the failure that forced the factor of two:

> at a 2.95 GB cache the chunk left 444 MiB and **about one conversation in three ended with no answer and no error**.

Two consequences for this milestone: (a) with borrow the cache cannot grow without bound - the planner's own rule
admits ~2.84 GB at chunk 8192 on this card (`free >= act*2 + head` with `free = 2210 - (cache - 2.2 GB)`); (b) the
silent-failure shape (no answer, no error) must be a gate, not a comment.

Also in that function, the blocker for overlap:

> Two buffers double-buffer the upload against the compute. **The borrow has only one range** (every layer takes the
> same bytes), so it is one buffer by construction.

So a borrow that keeps today's shape buys the memory back but **not the overlap**. Overlap needs either two disjoint
ranges in the loan or a ring (§6).

---

## 3. The numbers on this model (measured 2026-11-02)

`RUST_LOG=naiwa_rs::prefill_stream=debug` over one 1500-token prompt, cache 2.2 GB:

| per-layer expert stack | layers | layer ids |
|---:|---:|---|
| 650 MiB | 1 | 47 |
| 569 MiB | 6 | 17, 25, 27, 35, 40, 44 |
| 531 MiB | 11 | 0, 7, 14, 16, 18, 22, 31, 33, 37, 41, 46 |
| 481 MiB | 10 | 2, 9, 11, 15, 19, 23, 26, 32, 34, 39 |
| 456 MiB | 1 | 21 |
| 419 MiB | 3 | 12, 20, 30 |
| 369 MiB | 5 | 1, 8, 10, 13, 36 |

Same `[2560, 640]` shape, **seven dtypes**: `slot_bytes` (per expert, gate+up+down) is **1 881 600 B = 1.794 MiB** and
the cache holds **~26 slots/layer** at 2.2 GB. Free VRAM after dense weights, the drafter, KV and the cache:
**2210 MiB**, of which the plan spends 650 (one layer buffer) + 522 (activations, chunk 8192) + 512 (headroom).

Known caveat to fix alongside (affects any per-format reasoning): `provider.rs:349-352` sizes `slot_bytes` from
**layer 0** under a comment that asserts all layers share one shape, and then divides every layer's budget by it.
Layer 47's stack is 22% bigger than layer 0's, so its 26 slots overrun the per-layer budget by the same factor.

---

## 4. Why "borrow the slots of a layer with the same dtype" does not work (the earlier proposal)

The idea was: a prompt's per-layer volume goes into the cache slots that hold the **same quantization format**, and
formats without enough slots get a dedicated prefill buffer. The memory arithmetic on this model says no, because
the dtype groups are badly fragmented (table in §3):

| stack MiB | layers in group | slots the group can lend (26/layer) | MiB lent | MiB needed | short by |
|---:|---:|---:|---:|---:|---:|
| 650 | **1** | 26 | 47 | 650 | **603** |
| 569 | 6 | 156 | 280 | 569 | 289 |
| 531 | 11 | 286 | 513 | 531 | 18 |
| 481 | 10 | 260 | 467 | 481 | 14 |
| 456 | **1** | 26 | 47 | 456 | **409** |
| 419 | 3 | 78 | 140 | 419 | 279 |
| 369 | 5 | 130 | 233 | 369 | 136 |

A group is self-sufficient only when `cache >= 48 * stack / layers`: 2.26 GB for the 11-layer group, but
**21.4 GB** for the 456 MiB single-layer group and **30.5 GB** for the 650 MiB one. With dedicated top-ups the total
becomes **3 991 MiB** (cache 2.2 GB with 1 738 MiB of top-ups) against today's **3 552 MiB** - i.e. *worse*. Closed.

What survives from that proposal is the part that matters: the loan must be **priced in bytes and placed at the tail,
regardless of dtype** - which is what strata does with `slot_offsets` and we do not have at all.

---

## 5. The loan that works, priced

With the loan, the layer buffers no longer exist beside the cache, so the cache may take their bytes. What must stay
beside it is the chunk's scratch and the decode reserve, at the planner's own (measured) pricing:

```
beside the cache:  act * 2 (866..1044 MiB at chunk 8192) + head 512 = 1378..1556 MiB
free today at 2.2 GB cache: 2210 MiB   ->   cache may grow to ~2.8 GB
```

| shape | cache | loan | loan % of cache | beside the cache | total |
|---|---:|---:|---:|---:|---:|
| today | 2.2 GB | 0 (650 withheld beside) | - | 1684 | **3.86 GB** |
| loan, one range (what the flag does today) | 2.8 GB | 650 MiB | 23% | ~1556 | 4.36 GB |
| loan, **two** ranges (this milestone) | 2.8 GB | 1300 MiB | 46% | ~1556 | 4.36 GB |

- slots per layer: **26 -> 33** (+27%), free decode win of the same lever that measured TPOT 35.4 -> 33.4 ms at
  2.2 -> 2.5 GB (2026-11-01 sweep);
- overlap: the two-range loan is what M18 measured as **3.484 -> 2.830 s** on this exact 1500-token prompt (-19%);
- refill cost per long prompt: **650 MiB = 48 ms** (one range) or **1300 MiB = 97 ms** (two), i.e. 1.4-3.4% of that
  prompt's 2.8-3.4 s, paid once per prompt and only for the slots the loan covered;
- the loan never exceeds strata's 90% cap at this size (46%), so the "borrow almost everything" regime that slows
  their rich cards is not entered.

The loan's placement decides what the refill is *for*: slab banks are allocated layer-major, so a tail loan displaces
the **last ~12 layers** (one range) or **~24 layers** (two ranges). strata is laid out the same way and does not
reorder for it - see §10.1.

---

## 6. The plan

| step | change | where | size |
|---|---|---|---|
| **S1** | the slab is the default (no `NAIWA_CACHE_SLAB`), sized `--expert-cache-gb`; every bank inside it, **each slot of its own layer's size, 256-B aligned, back to back, with an offset table** (strata's `open_sized`/`slot_offsets`) - which also removes the "slot_bytes from layer 0" assumption of §3 | `provider.rs:83-111`, `slab.rs:81` | S |
| **S2** | the loan moves to the **tail** and holds **two** disjoint ranges (A/B for the double buffer), addressed through the offset table rather than a byte cursor | `prefill_stream.rs:477-620`, `slab.rs:123` | M |
| **S3** | `bytes_needed(buffers, ring_slots, chunk)` as a pure function (the ring term 0 for now, §10.3) + the two caps (`k + 128 <= slots`, `k <= 90%`), chosen **per request**; `stride = 2` allowed when borrowing | `prefill_stream.rs:110-192` | M, unit-testable |
| **S4** | the ledger: `Vec<(layer, slot)>` for the loan, refilled **queued with one wait** by the copy stream under `ACCESS`, then a byte-for-byte `verify_slot` of one refilled slot | new, `staging.rs` | M |
| **S5** | the planner's side becomes "beside the cache: `act*2 + head + decode reserve`", and the silent-failure shape becomes a counted error | `prefill_stream.rs:150-192` | S |
| **S6** *(optional, later)* | per-expert ring in the loan (strata's part 5), which needs blob-ordered banks and buys resident-skipping (7-9%) | large; separate milestone | L |

Deliverable of this milestone is **S1-S5**: the cache back to ~2.8 GB, two ranges, a refill that returns the loan, no
new kernel, no new stream, no change to the numeric path.

---

## 7. Rejected alternative: free the cache and allocate it at decode start

Raising the same idea as allocation rather than aliasing. It is mechanically possible - candle allocates through
`cuMemAllocAsync` on the stream's pool (`cudarc-0.19.10/src/driver/safe/core.rs:1628-1632`), so both alloc and free
are stream-ordered and non-blocking - but it loses on three points and wins nothing over §5:

1. **Residency, every turn.** Freeing the cache drops the slot contents; at 2.8 GB the rebuild is 2.8 GB of PCIe
   (0.21 s) plus the cold-miss burst. In the recorded chat run the post-turn tails are 23-108 new tokens against a
   `--prefill-stream-min-tokens` of 64, so most turns would switch phases twice and pay it twice.
2. **Addresses move.** The mapped routing lists and the lookahead hold device pointers into cache slots; a realloc
   invalidates them. The slab keeps one address for the session - that is the whole reason `Slab` exists.
3. **Pool churn.** 48 layer allocations of ~46 MiB against freed 650 MiB blocks means splitting and a pool that
   keeps the tail; the recorded history of lending is exactly a 400 MiB leak from repeated lend/return.

The alias reaches the same memory profile with **zero** allocator calls and keeps residency. Recorded here so the
idea is not re-derived.

---

## 8. Risks (all of them already met once)

1. **Fills writing into the loan while a chunk computes -> NaN.** Measured in M18: 62 non-finite chunk outputs per
   run at `--pcie-frac 0`. `BORROWED` + `ACCESS` + `quiesce` fixed it; the **refill** is a new writer and must take
   the same lock.
2. **Sizing the loan per bank instead of by bytes.** History: a 1 GB cache lent 1107 MiB of a 1300 MiB need.
3. **Lend/return churn.** 400 MiB leaked before. The loan's position is fixed; only its length changes; no
   alloc/free per request.
4. **`borrow_from`'s pointer is not tied to the copy stream** (`device_ptr_with_guard` ignores it - the pending
   upstream PR 1). Correctness leans on explicit event ordering, as today.
5. **Silent capacity failure.** The 2.95 GB arm produced "no answer and no error" for ~1/3 of conversations. A gate,
   not a comment.

---

## 9. Gates

| gate | criterion |
|---|---|
| numeric | `frac 0` + `--no-lookahead` **byte-identical** to `harness/m1/reference.oracle.txt` |
| no NaN | non-finite chunk outputs = 0 (M18's counter) |
| the loan is real | one 1500-token prompt, wall clock: today 3.484 s, `NAIWA_STREAM_SYNC=1` 3.331 s, M18's two-buffer 2.830 s -> target **< 3.0 s** |
| the refill happens | counters: slots lent vs bytes refilled; after a prompt the cache is back to its pre-prompt slot count |
| decode not worse | chat `short`, 3 rounds: TPOT p50 not above the 43 ms of the shipped config, and no spike on the first step after a long prompt |
| tail TTFT | chat `short`: `TTFT>=2` p50 moves toward strata's 651 ms (this is the parent lever; not a gate on this milestone) |
| units | `bytes_needed` / lend cap / `stride` decision tests, in the style of `plan_spends_memory_on_the_chunk_before_the_second_buffer` |
| the slot table is true | one refilled slot read back and compared to its host blob, byte for byte (strata's `verify_slot`: *"the only thing that says the cache holds the expert it claims to"*) |
| silent failure | a cache too big for the loan is a **reported** refusal, and the run still answers |

---

## 10. The three decisions, taken from strata

### 10.1 Where the loan sits: the slab tail, layer-major, no heat re-ordering

strata's cache is layer-major by slot index - `ExpertCache::layer_slot_range` (`expert_cache.cpp:624-631`) is
`q = slots / n_layers`, layer `l` owns `[l*q, (l+1)*q)`, and the **last** layer takes the remainder up to `slots`.
So "the last k slots" **is** "the last layers' slots", exactly what a our slab tail gives us; the heat ranking
(`rank_learned_profile`, `expert_cache.cpp:143-166`: resident first, then heat desc, then the pack's prior) decides
**which experts fill a layer's slots**, never where the slots are. No reordering work, and the refill cost lands on
the last layers - which is also where our profile warmup already puts the least-used ones.

The same code answers the layout question: a pack whose expert blobs differ per layer is `open_sized`
(`expert_cache.cpp:561-583`) - **one arena, slots of each layer's own size, 256-byte aligned, back to back, plus an
offset table** (`slot_offsets()`, null when uniform), and each layer's admission cursor starts at the bottom of its
own range. That is the fix for our layer-0 `slot_bytes` assumption (§3) and the address map S2 needs.

### 10.2 Refill: eager, queued, one wait - and verified

A prompt's loan is given back **immediately after the prompt**, as queued copies followed by one wait:
`fill_slot_queued` + `sync_queued` (`expert_cache.hpp:196-200`, perf-review D-4: *"many slots are refilled with one
`sync_queued` at the end instead of a wait per slot"*), driven from `generate.cpp:9570-9600`. The per-slot blocking
form is the diagnostic arm only (`STRATA_REFILL_BLOCKING=1`), and a refilled slot is checked with `verify_slot`
against its host blob. So: not lazy, not per-slot waits, and a byte check - our `CopyStream` plus one synchronise is
the same shape.

### 10.3 The ring is **not** in this milestone - but S3 prices it

strata's history has the loan first and the streamed ring later: `generate.cpp:3207-3209` notes that its reserve
estimate "predates the streamed ring", and the ring only appears from the 1024-token chunks of #340. Its loan
pricing is one function that already includes the ring term (`Prefill::bytes_needed`), which is why S3 takes
`ring_slots` as an argument from the start (0 now) - the loan and its cap then need no rework when the ring lands.
The ring itself is the large change (blob-ordered banks, §6 S6) and gets its own milestone.

---

## 11. What was built, and what the first measurement says (2026-11-02)

### Built

| step | what landed | where |
|---|---|---|
| **S1** | The cache's banks are carved from **one allocation** ("the lending window"), on by default (`NAIWA_CACHE_SLAB=0` is the pre-M33 layout). The window is sized to **two layer buffers** (1536 MiB) and falls back by halves on a refusal, because at `--expert-cache-gb 2.5` this card refuses the 1.5 GiB block outright - which cost a whole chat run its stream before the fallback existed. | `provider.rs` `slab()`, `staging.rs` |
| S1b | The slab records its **layout** (`Range { offset, bytes }`, `overlapping`, `tail_start`), i.e. strata's `slot_offsets`: the loan needs to name the banks a byte range covers. Unit-tested without a device. | `slab.rs` |
| **S2** | The loan holds **two disjoint ranges** (`buf * span`), so the double buffer exists when the stream borrows - the tree before this had "one range by construction" and therefore no overlap by construction. The clearing and the zeroing cover both ranges. | `prefill_stream.rs` `borrow_buffers` |
| S2b | The two per-layer barriers (`after_device_work`, the copy-stream `synchronize`) became **once per chunk**. As per-layer calls they ordered layer L+1's prefetch behind layer L's own kernels, i.e. behind exactly the work it was meant to overlap. | `prefill_stream.rs` |
| **S3** | `plan_with_lend(free, layer, ceiling, borrowing, kv_reserve, lend_bytes)`: with the loan, the buffer count follows the **pool** instead of the free VRAM. Unit tests pin both arms. | `prefill_stream.rs` |
| **S4** | The **ledger**: `empty_range` records every `(layer, expert)` it displaces (`Layers::lent`) and the first decode step after a chunk refills them (`Layers::refill_lent`, one queued copy each, no wait - strata's `fill_slot_queued`). Logged as `the loan is given back, N of N displaced experts queued for refill` (719 of 719 on a 1239-token prompt). | `cached_layer.rs` |
| S5 | The borrow reads `NAIWA_PREFILL_BORROW` as a **value** (`=1` on, anything else off) instead of presence - the presence test made `=0` mean *on*, and the first A/B of this milestone ran the loan twice. **The default stays off**, for the reason in §11.5. | `slab.rs` |

### Measured

Gates first: `frac 0` + `--no-lookahead` is **byte-identical** to `harness/m1/reference.oracle.txt` (three captures
during the work), `cargo test` 54 passed, and `NAIWA_BORROW_CHECK=1` found no non-finite output in any arm.

**1. The loan works as designed - and the double buffer is real.**

| arm (chat config, 1239-token prompt, cold process, 3 interleaved rounds) | chunk wall | buffers |
|---|---:|---|
| `NAIWA_PREFILL_BORROW=0` | 3.791 / 3.769 / 3.762 (median **3.769** s) | 1 |
| `NAIWA_PREFILL_BORROW=1` | 3.840 / 3.606 / 3.592 (median **3.606** s) | **2** |

-4.3%, and the first-token text is identical in both arms, which is what says the chunk's math is unchanged (the
prefill path never reads the cache's slots, so a correct loan cannot move it).

**2. The 19% M18 measured is not there on this workload, and the reason is arithmetic.** M18's two privately-allocated
buffers read 2.830 s on the same prompt. What a second buffer can hide here is the per-layer *compute*, and MoE
prefill is weights-bound: a layer is 650 MiB of weights against ~1240 rows of activations, i.e. ~1-2 ms of compute
behind ~50 ms of copy. The 3.6 s we measure is also **not** the link: 23.4 GiB at 13.4 GB/s is 1.87 s, so half the
chunk is per-layer latency, not transfer. That gap, not the overlap, is the next prefill lever.

**3. The prize was capacity, and it is half-blocked.** Decode TPOT tracks the cache closely, and the loan is what
lets the cache grow *without* withholding the chunk's buffers:

| cache | loan | loads? | long prompt | TPOT (median of 5 for the 2.2/2.5 rows) |
|---:|---|---|---|---:|
| 2.2 GB | off | yes | 3.27 s | 36.74 ms |
| 2.2 GB | on | yes | 3.27 s | 36.74 / 35.15 ms (see below) |
| 2.5 GB | on | yes | 3.14 s | 35.15 ms |
| 2.9 GB | on | yes | **500** | 33.93 ms |
| 3.2 GB | on | yes | **500** | **28.59 ms** |

So 2.2 -> 3.2 GB is **-22% TPOT** (36.7 -> 28.6 ms) *if* the prompt still runs - and it does not, because the
chunk's **scratch** (activations, `act * 2 + head`) is still paid from free VRAM beside the cache. The loan covers
the layer buffers only; strata's `bytes_needed`/`relayout` price **every** buffer of the chunk. Putting the chunk's
scratch in the pool too is therefore the step that buys the remaining ~7 ms, and it is the natural S6.

**4. And it fixes M23's failure mode.** With the loan the plan stops charging a 650 MiB buffer against free VRAM, so
the "after a warmup conversation 1 787 MiB is left, one buffer does not fit, the stream turns itself off and the turn
fails" path (the reason `engines/naiwa.json` is pinned at 2.2 and not 2.3) cannot trigger. That is why 2.5 GB was
tried at all - and why it was **not** shipped: at 2.5 the 1.5 GiB window did not fit, the fallback did not exist yet,
the loan silently went away, and the M23 refusal came back and failed a turn. With the fallback in place the shipped
config stays at 2.2 GB; raising it is the follow-up, once the chunk's scratch is in the pool.

**5. On the shipped chat config the loan is net negative, and that is the result.** Same session, same config,
`short`, 3 rounds, 8 measured turns each, 0 failed turns in both arms:

| arm | TTFT t1 | TTFT >= 2 p50 | p90 | TPOT p50 | tok/s | e2e per conversation |
|---|---:|---:|---:|---:|---:|---:|
| loan **off** (shipped) | 942 ms | 1 105 ms | 2 264 ms | 43 ms | 23.4 | **30.2 s** |
| loan **on** | 955 ms | **969 ms** | 2 239 ms | 43 ms | 23.1 | **39.2 s** |

The loan buys the tail (-136 ms of TTFT p50, -12%) and costs **+9 s per conversation (+30%)**. The mechanism is
one loan per conversation, not per turn (the logs show exactly one `the loan is given back` per round), and it
displaces **719 of 1 474 slots** - half the cache - at the turn-1 chunk. Refilling those is 1.5 GB of PCIe, and the
conversation's reuse then works against a cache whose residency was reset. This is strata's own warning in their
source ("A LOAN IS NOT FREE PER REQUEST ... a 2K prompt read 37% slower on cards that hold nearly every expert of
their layers"), met on a card that holds 10% of them.

What would change the verdict is what §11.3 and §11.4 point at: a loan that is **sized to the request** (strata's
`relayout`, which lends only what a prompt needs - our buffer is a whole layer, so a small loan needs the per-expert
ring) and a cache that **grows into the bytes the loan frees** (S6). Until then the machinery stays behind
`NAIWA_PREFILL_BORROW=1` and the shipped path is the pre-M33 one, with the window in place and unused.

### Where this leaves M33

- The mechanism is in, it is numerically clean (byte-exact oracle, no non-finite output, 54 unit tests), it needs no
  VRAM, and it gives the plan a double buffer it could not afford otherwise.
- On the config that ships it is **net negative** (+30% e2e for -12% tail TTFT), so it stays opt-in and the milestone
  stays open rather than done.
- What it proved is where the value actually is: **the cache's size**, worth **-22% TPOT at 3.2 GB** (36.7 -> 28.6
  ms), which is blocked by the chunk's *scratch* still being paid from free VRAM. That is S6 - put the chunk's
  activations in the pool like strata's `relayout` prices every buffer of a chunk - and it is worth more than the
  loan itself.
- Two smaller findings to keep: the prefill is **1.87 s of copy in a 3.6 s chunk**, i.e. half of it is per-layer
  latency rather than transfer (an M23-shaped lever); and the window's allocation can be refused outright at a bigger
  cache, which is why it now falls back by halves instead of taking the stream down with it.

---

## 12. S6-lite: pricing the chunk by what it measures, and the ceiling that came out of it

S6 was "put the chunk's scratch in the pool like strata's `relayout` prices every buffer of a chunk". The cheap
version - S6-lite - was to price the chunk honestly and let the plan pick a size that fits, so a bigger cache could
be used. What the measurements say is: the price was wrong, the pricing is now right, and **the bigger cache still
does not fit** - for a reason that has nothing to do with the loan.

### 12.1 How it was measured

`NAIWA_PREFILL_PEAK=1` already logged a chunk's peak above its own baseline. It now also logs the **per-layer free
VRAM trace** and the **live/retired bytes of the MoE workspaces** (`mistralrs_quant::gguf::cuda::moe_workspace_bytes`,
new). Same load, same chat config, one long request, `--expert-cache-gb 2.2`, loan on:

```
prefill peak (low water at layer 47): chunk 1216 rows, 48 layers, 1730 MiB used
  (2113 MiB free at start, 383 MiB low water); layer buffer 1300 MiB, planner estimate 1034 MiB (act 522)
prefill peak trace (layer: MiB used): 0:0 1:1090 2:1122 3:1506 ... 7:1570 ... 35:1666 ... 47:1730
prefill peak: MoE workspaces 0 MiB live + 0 MiB retired
```

Three things fall out of that one line and one repeat at 288 rows:

| rows | peak | peak − 384 MiB KV reserve |
|---:|---:|---:|
| 288 | 1 314 MiB | 930 MiB |
| 1 216 | 1 730 MiB | 1 346 MiB |

1. **The chunk's footprint is not proportional to its rows.** Four times the rows bought 416 MiB, i.e.
   **0.448 MiB a row over a fixed 801 MiB**. The old estimate (`chunk * 24 KiB` + a per-batch MoE term) called it
   348 MiB for 1 216 rows: **off by 4x**, and in the dangerous direction.
2. **The fixed part is one-time, not per chunk.** It appears in big steps at the first prompt compute
   (layer 1: +1090, layer 3: +416 more, then a few +64/+96) and never comes back, which is why free VRAM after a
   long prompt stays low and why decode's reserve matters. Shrinking the chunk does not shrink it.
3. **It is not ours.** The MoE workspaces the fork keeps per (device, stream) - with a `retired` list that never
   frees - measure **0 MiB** for this model. That hypothesis is dead, cleanly, which is what the probe is for.

### 12.2 What changed

`chunk_scratch(rows) = 801 MiB + 0.448 MiB * rows`, floored by the old shape estimate, both terms overridable
(`NAIWA_CHUNK_FIXED_MIB`, `NAIWA_CHUNK_PER_ROW_KIB`), and **pinned by a unit test to the two measured points**
(`the_chunk_line_is_the_two_points_that_were_measured`: 288 -> 930, 1 216 -> 1 346). The borrow arm of the plan uses
it plus the 512 MiB headroom instead of the `act * 2` fudge; the owning arm is untouched, because its behaviour is
what the shipped config was measured with. `log_plan` and the "stream is off" warning now print **what the plan
charged**, not the owning arm's sum - the old line read "2 172 MiB of 2 113 MiB free" on a plan that had been
accepted, which is how a plan loses its reader.

The line reproduces the measurement it came from: at 2.2 GB the chunk used 1 346 of the 1 729 MiB left after the KV
reserve and the probe read 383 MiB of low water - exactly the difference.

### 12.3 The ceiling, and why 3.2 GB is not reachable this way

The plan is honest now, so it refuses instead of dying mid-chunk. What it refuses at:

| cache | free before the first chunk | needs (chunk scratch + head + KV) | outcome |
|---:|---:|---:|---|
| 2.2 GB | 2 113 MiB | 1 346 + 512 + 384 = 2 242 (a 768-row chunk: 1 640) | runs, 383 MiB of low water |
| 2.5 GB | 1 793 MiB | 1 346 + 512 + 384 = 2 242 | 768-row chunk planned, and the **second** chunk OOM'd in `moe experts forward` with 191 MiB left |
| 2.9 GB | 1 505 MiB | floor is 1 697 | refused at the plan; with the old optimistic price it used to die inside the experts forward (HTTP 500) |
| 3.2 GB | ~1 305 MiB | — | does not start |

So the honest statement, replacing the one in §11: **the cache cannot exceed ~2.2 GB while a prompt has to stream**,
because the chunk needs ~2.0 GiB of free VRAM and a chunk is required for every prompt of 64 tokens or more (i.e.
most continued turns). The 28.6 ms TPOT measured at 3.2 GB belongs to requests that never streamed (short prompts),
not to a config that works.

Two blocks hold that ceiling, and both are now measured rather than suspected:

- **the chunk's fixed 801 MiB** - inside the model's kernels, not in this crate and not in the fork's workspaces;
- **the MTP drafter's ~900 MiB** (free VRAM at the same point is 3 201 MiB without `--mtp` against 2 305 with it).

Trading the drafter for the cache is the wrong trade by the numbers already recorded: MTP is worth ~8 ms of TPOT
(51 -> 43), and 2.2 -> 3.2 GB is worth ~8 ms at best while it breaks long prompts outright.

### 12.4 Where the next real lever is

1. **The chunk's fixed 801 MiB**: it is a handful of one-time blocks in the prompt compute. Finding them is a
   `nsys`/allocation-trace job, and it is the only thing between this card and a 3 GB cache.
2. **Per-layer slot sizing** (the open half of S1): `slot_bytes` is priced from layer 0, so a 2.147 GB slab filled
   after 129 of 144 banks and 15 spilled - the cache is ~10% smaller than its nominal size at the same VRAM. Fixing
   it is pure bookkeeping in `provider.rs`/`slot.rs` and buys ~0.2 GB of residency for free.
3. **The prefill's 1.87 s of copy in 3.6 s of wall** (§11.2): half a chunk is per-layer latency, which is an M23-shaped
   lever and independent of all of the above.

---

## 13. The loan never lent: what `NAIWA_BORROW_ALIAS` was costing, and the corrected chunk line (2026-11-03)

### 13.1 The defect

`borrow_buffers` reached the slab - the whole point of the milestone - only when `NAIWA_BORROW_ALIAS` was set:

```rust
if std::env::var_os("NAIWA_BORROW_ALIAS").is_some() {
    ... slab.borrow_at(offset, ...)          // the pool's bytes are the buffer
} else {
    // Probe: keep everything else ... but give the chunk memory of its own
    let mem = self.copy.alloc_like(stack)?;  // <- the default, and an allocation beside the cache
}
```

So the default arm did **all of the loan's bookkeeping and none of its lending**: the plan charged the chunk as if the
layer buffers were free, the cache's tail was emptied (719 slots) and refilled afterwards, and the memory still came
out of free VRAM. Both of §11's chat arms, and every probe behind §11 and §12, ran that way.

One request, one load, only that switch changed (`NAIWA_PREFILL_PEAK=1`):

| rows | chunk peak, private buffers (the default) | chunk peak, pool ranges (the loan) |
|---:|---:|---:|
| 288 | 1 314 MiB | **66 MiB** |
| 1 216 | 1 730 MiB | **386 MiB** |

What the two "private" numbers actually contain is one or two **650 MiB layer buffers**; the chunk's own footprint is
the difference, and it is **~0.35 MiB a row with no meaningful fixed term**. The shape estimate the plan started from
(`24 KiB a token`) was 14x short, and §12's replacement (`801 MiB + 0.448 MiB a row`) was 3.5x long - it had absorbed a
privately allocated buffer into what it called the chunk.

### 13.2 What changed

- **The loan lends by default.** `NAIWA_BORROW_PRIVATE=1` is the beside-the-cache arm (what M18 measured);
  `NAIWA_BORROW_UNUSED=1` stays as the in-slab probe.
- **`chunk_scratch(rows) = 32 MiB + 354 KiB * rows`**, pinned by unit test to both measured points (`131` and `452`
  MiB, a few percent over 66/386 because the probe samples per layer and a chunk's transients live between samples).
  The `max(activation_bytes(...))` floor was **removed**: the shape estimate is 337 MiB at 288 rows, so a floor would
  price a small chunk 5x too high - which is exactly how the first attempt refused a 3.2 GB stream that then ran.
- **The lent arm prices `chunk_scratch + 256 MiB`** (`CHUNK_HEAD`), not `+ 512`: the measured peak already carries the
  growth the "head" used to stand in for, and charging both refused a 1 024-row chunk at 3.2 GB that ran at 386 MiB of
  961 free. The owning arm keeps the 512 MiB block and now pays the measured line too.
- **Consequence for the double buffer**: with a row-priced chunk, the owning arm's "the same chunk must still afford
  the second buffer" condition becomes `free >= free + 650 MiB`, i.e. never on this card. That is the honest answer -
  the plan has always preferred rows to a second buffer, and M33 §11.1 measured the second buffer at -4.3% against
  ~1 900 more rows. The double buffer is still granted where it is free (the loan, and an 8 GiB card).

### 13.3 Measured

Gates: `frac 0` + `--no-lookahead` **byte-identical** to `harness/m1/reference.oracle.txt`; **55 tests** pass;
`NAIWA_BORROW_CHECK` clean.

**The chunk, with the loan, at every cache size that has room:**

| cache | first-chunk plan | peak | long prompt (1 216 tokens) |
|---:|---|---:|---:|
| 2.2 GB | 4 096 rows, 2 ranges | 386 of 2 113 MiB | 3.90 s |
| 3.2 GB | 768 rows, 2 ranges | 386 of 959 MiB | **3.67 s** |

So `--expert-cache-gb 3.2` runs a long prompt, which §12 recorded as impossible. The chunk is no longer what stops it.

**But the loan is still not the shipping default, and now for the measured reason.** Chat harness, `short`, 3 rounds,
8 measured turns each, shipped 2.2 GB config:

| arm | failed turns | TTFT >= 2 p50 | TPOT p50 | e2e per conversation |
|---|---:|---:|---:|---:|
| loan off (shipped) | 0 | 1 083 ms | 43 ms | **36.4 s** |
| loan on, run A | **1** | - | - | - |
| loan on, run B | 0 | 915 ms | 45 ms | 38.8 s |

The wipe is the cost: one loan per conversation still displaces 719 of 1 310 slots, and at 2.2 GB the cache is worth
more to decode than a free layer buffer is to prefill. Run A's failure is the M23-class silent one ("no content delta,
finish None"), with nothing in the log - it lands right after `the loan is given back`, so the refill path is the
suspect, and 1 in 72 turns is enough to keep the loan off by default until that is understood.

### 13.4 And the cache ladder does not reproduce

Same session, `harness/tpot` 5 rounds, loan on, honest pricing:

| cache | TPOT rounds | median | long prompt |
|---:|---|---:|---:|
| 2.2 GB | 48.4 / 41.4 / 34.1 / 43.9 / 47.8 | 43.9 ms | 3.90 s |
| 3.2 GB | 55.2 / 39.0 / 47.0 / 36.9 / 45.0 | **44.98 ms** | 3.67 s |

**The -22% TPOT that M33 §11.3 recorded for 2.2 -> 3.2 GB (36.7 -> 28.6 ms) is withdrawn.** Today's repeat measures
+1.1 ms, and the spread inside a single load (34-55 ms) is larger than any cache-size effect this instrument can
resolve. Two things are true at once: the cache genuinely matters to decode (the T0 slope, 0.4-10.7 ms/GB depending on
`--pcie-frac`), and this card/config cannot resolve a 1 GB step with 5 rounds of a single-request probe. The chat
harness is the instrument that can (24 turns, per-turn median), and it says the loan is not free at 2.2 GB.

There is a second, structural reason the ladder cannot be trusted yet: **`slot_bytes` is priced from layer 0**, so the
nominal cache is not the resident one. Measured: a 2.147 GB slab filled after **129 of 144** banks and spilled 15. At
3.2 GB that overshoot costs ~10% of the budget, so a "3.2 GB" cache is really ~3.0 and the step between two rungs is
partly an accounting artifact.

### 13.5 Where this leaves the milestone

- The mechanism is correct now, and it is the only way this card can stream a prompt without paying 650-1 300 MiB for
  the privilege: **386 MiB instead of 1 730** for a 1 216-token chunk, and a 3.2 GB cache that starts.
- It stays opt-in because the wipe costs more than the buffer it frees at 2.2 GB, and because one run in two produced
  a silent empty answer right after a refill.
- Two things must happen before a cache ladder means anything: **per-layer slot sizing** (so nominal == resident) and
  the refill's silent-failure path understood. Then the ladder is worth running again, on the chat harness, with the
  loan, and the shipping cache size can be chosen from numbers that repeat.
