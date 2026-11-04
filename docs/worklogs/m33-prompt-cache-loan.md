# M33 - the prompt path borrows the expert cache (strata's loan)

**Status:** open; S1-S4 built, plus **S6-lite (§13) and the per-layer slot price (§14)**; §12 and §13 are two rounds of the same investigation and they **contradict each other
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

---

## 14. The cache's size was a lie of ~16%: per-layer slot pricing (2026-11-03)

§13.4 blamed the cache ladder on `slot_bytes` being priced from layer 0. That was the right suspect for the wrong
reason: the price is computed per layer already, but **from one projection**. `slot_bytes_from_qtensor(gate) * 3`
assumes gate, up and down share a quant, and in this model they never do:

| what the GGUF says (parsed 2026-11-03) | |
|---|---|
| layers whose three projections differ in format | **48 of 48** |
| typical layer | `ffn_down_exps` IQ4_NL, `ffn_gate/up_exps` IQ2_S |
| real bank / (`3 x gate`) | **0.885 - 1.252** |

So the budget was charged 12-25% too little on most layers, and the cache quietly kept more experts than it was
told it could. Measured consequences before the fix: a nominal **2.2 GB** cache held **1 310 slots** (~2.55 GB of
experts) and left 2 369 MiB free; a 2.147 GB slab filled after **129 of 144** banks and spilled 15.

The fix is `slot_bytes_from_stack`: the three projections, each in its own dtype, summed - the same number the bank
actually allocates. Pinned by `slot_bytes_price_each_projection_in_its_own_format` (a mixed stack must cost more than
`3 x gate`; a uniform one must cost exactly that).

### 14.1 What it changed

| | before | after |
|---|---:|---:|
| slots resident at a "2.2 GB" cache | 1 310 | **1 145** |
| free VRAM after load (chat config) | 2 369 MiB | **2 687 MiB** |
| nominal cache that holds the same residency | 2.2 GB (really 2.55) | **2.5 GB** (really 2.5) |

165 fewer slots and 318 MiB more free VRAM at the same nominal size is the overshoot, recovered. The shipped config
moved to **2.5 GB** so the number means what it says and the residency the previous pin was actually getting is kept.

### 14.2 Measured, chat harness, `short`, 3 rounds, 0 failed turns everywhere

| config | TTFT >= 2 p50 | TPOT p50 | e2e per conversation |
|---|---:|---:|---:|
| 2.2 GB, old pricing (really 2.55 GB, 1 310 slots) | 1 083 ms | 43 ms | 36.4 s |
| 2.2 GB, honest pricing (1 145 slots) | 1 091 ms | 46 ms | 39.8 s |
| 2.5 GB, honest pricing (same residency as the old 2.2) | 1 107 / 1 021 ms | 42 / 45 ms | 43.5 / 37.4 s |
| 2.5 GB, shipped, repeat | 1 107 ms | 42 ms | 43.5 s |

Two things to read out of that. The overshoot was worth ~3 ms of TPOT at a fixed nominal size, which is the price of
an honest budget and is now paid knowingly. And **e2e varies 36-43.5 s across loads with the same config**, so it is
not an instrument for a 1-3 ms TPOT question - TPOT is (42-46 ms across configs, its spread inside one load is
±5 ms).

### 14.3 State of the milestone

- The chunk is priced by measurement (0.35 MiB a row), the loan lends by default when asked for, and the cache's
  nominal size is now the resident one.
- What is **not** established: that any particular cache size is faster. The ladder measurements contradict each
  other run to run (43.9 vs 44.98 ms for 2.2 vs 3.2; 42 vs 46 for 2.5 vs 2.2; the historical 36.7 -> 28.6 is
  withdrawn), and the honest conclusion is that this card/config does not resolve cache-size steps with the
  instruments in the harness today.
- What is left, in order: (1) the refill's silent empty answer (§13.3, 1 turn in 72 - a correctness question, not a
  performance one); (2) a TPOT measurement that can resolve a cache step - e.g. many rounds of one fixed prompt with
  the *same* load, which removes the per-load variance, instead of reloading per arm; (3) only then the cache default.

---

## 15. The silent failure is a NaN, it belongs to the loan, and it is not the refill (2026-11-03)

§13.3 left "one chat run in two refilled into a silent empty answer" as a correctness question. It is now three
things: a *visibility* fix, a *diagnosis*, and two eliminated hypotheses.

### 15.1 It was never silent - the harness was

The server answered `{"error": {"message": "Internal server error.", ...}}` on the SSE stream (and as a 500 body when
it failed earlier), and `harness/chat/run.py` dropped any payload it did not recognise as a delta or a finish reason.
So a failed request surfaced as `no content delta (finish None)` with nothing to act on - the harness was the silent
half. It now raises on an `error` payload, keeps non-SSE lines, and records them as `server_said`, so the same failure
reads `server error: {...}` and names itself.

With the reason visible (`NAIWA_LOG_INTERNAL=1`, a hook that already existed for exactly this), the failure is:

```
ERROR mistralrs_server_core::handler_core: internal error, client sees the generic message:
      invalid CUDA top-1 output: max_logit=NaN argmax=NaN
```

**The loan puts NaN in the logits.** Not an OOM, not a refused stream, not an empty answer - the top-1 sampler refuses
to sample a non-finite logit, and the request dies.

### 15.2 What the probes rule out

| probe | result | what it rules out |
|---|---|---|
| 5 chat runs, loan **off** (120 turns) | 0 failures | that the failure is the chat path's baseline flakiness |
| 4 chat runs, loan **on** (96 turns) | 4 failures, every run | that it is unrelated to the loan |
| `NAIWA_LOG_INTERNAL=1` | `max_logit=NaN` | an OOM, a plan refusal, a swallowed engine error |
| `NAIWA_PREFILL_PEAK` (`PREFILL_NAN`, new) | never fires | a chunk whose **inputs** are already garbage; the hidden states entering every layer are finite |
| `NAIWA_BORROW_CHECK=1` (`DECODE_NAN`, existing) | `DECODE_NAN layer 0 (rows 14)` | that it starts in the dense part: the *first* layer's MoE output is already non-finite |
| refill given one `synchronize` (`sync_queued`) | still NaN | that the refill's copies were being read in flight - the fix is right and stays, but it is not the cause |
| `NAIWA_REFILL=0` (new) | **still NaN** | the refill entirely: a loan with no refill at all still corrupts the logits |
| `NAIWA_PREFILL_QUIESCE=1` | still NaN | lookahead/prefetch work in flight during the chunk |

So the corruption is in the **loan itself**: a chunk writes into the cache's bytes, and afterwards something reads a
slot whose contents the chunk overwrote - or the chunk's own compute reads a buffer that is not what was uploaded.
`empty_range`'s `BORROW_STILL` never prints, so it is not the "a slot still claims an expert" defect it was written
for.

### 15.3 Where the next probe has to look

The two candidates left, and the probe for each:

1. **A slot published after the chunk.** `empty_range` clears `slot.expert` twice (phases 1 and 3) precisely because a
   fill reserved before the borrow can publish after it. `admit`/`load_reserved` check `slab::borrowed()` under
   `lock_access` - so the question is whether some *other* path publishes a slot or a route entry without that gate
   (the mapped route table, the lookahead's publish, the promotion of a staged expert).
2. **The chunk reading the wrong bytes**, i.e. `borrow_at(offset)` landing on memory the pool does not own for that
   layer (the second range is at `span = layer_bytes`; the pool holds two of them and 236 MiB besides). Strata's
   `verify_slot` exists for this: after an upload, compare the device range against the host stack.

Either way the loan stays **off by default**, which is where it already is.

---

## 16. A cache ladder that can answer the question (§13.4 said the old one could not)

`harness/tpot/ladder.py`, plus a small change to `serve`: it now shuts down gracefully on SIGTERM/Ctrl-C, and with
`NAIWA_STATS=1` prints the process's expert-cache counters at exit. The script runs **one cold load per rung** with
**many rounds on that load** and reports the median with the *within-load* interquartile spread, next to the
mechanism's own numbers.

Measured on the shipped config (loan off, 1 216-token prompt, 96 tokens out):

| cache | TPOT p50 [p25, p75] | decode tok/s | decode hit rate | PCIe MiB/round |
|---|---|---|---|---|
| 2.2 GB | 33.78 ms [32.61, 41.00] | 29.6 | 65.9% | 16 695 |
| 2.5 GB | 32.83 ms [31.68, 35.83] | 30.5 | 68.5% | 16 059 |
| 2.9 GB | 33.71 ms [32.46, 35.01] | 29.7 | 70.8% | 17 041 |

and the repeat (2.2 vs 2.9, 10 rounds): 34.35 [32.75, 38.91] vs 35.45 [34.39, 36.36), i.e. **+1.10 ms against a
6.16 ms spread**.

**The ladder is flat.** 0.7 GB more cache buys 5 points of hit rate and no time. That is the answer the previous
instrument could not give, and it says the same thing the mechanism does: at these sizes the cache is not the
constraint on this workload, so **the cache size is not where the remaining time is** - and the earlier "2.2 -> 3.2 GB
= -22% TPOT" is now not merely unreproduced but contradicted by a measurement with error bars around it.

Two by-products worth keeping: a rung's TPOT is **bimodal** (33 ms typical, 93-101 ms for a round that re-prefills),
which is why the protocol's median is the right statistic and a mean is not; and the counters show **~16.6 GiB of PCIe
a round**, which is the prompt's own 23.4 GiB of layer uploads - i.e. the prompt path, not decode, is what moves bytes
on this card.

---

## 17. The NaN is found and fixed: the loan emptied its slots once per *allocation*, not once per chunk (2026-11-03)

§15 narrowed the NaN to "the loan corrupts memory in a way neither the refill nor the quiesce reaches". It is now
found, and it was a scope mistake in this milestone's own code.

### 17.1 How it was pinned

1. **`NAIWA_BORROW_VERIFY=1`** (new, strata's `verify_slot`): after a layer's upload lands, read the device range back
   (`QTensor::data()` on a CUDA tensor is a d2h of the whole projection) and compare it byte for byte with the host
   stack. It ran on layers 0 and 32 of every chunk and found **zero differing bytes, zero length mismatches**. So the
   chunk is computing with exactly the weights it uploaded: the corruption is on the **read** side.
2. The router only accepts a hit when `slot.expert == Some(id) && slot.buf.is_some()`, so a hit cannot point at the
   wrong id. What is left is a slot whose `expert` is set while its **bytes** belong to something else - i.e. the
   chunk wrote over a slot that was still resident.
3. `empty_range` is what prevents that, and it was reachable only from `borrow_buffers`... which runs only when the
   layer buffer is **(re)allocated**. A chunk whose shapes match the ones already in `buf.memory` skips
   `borrow_buffers` entirely - and with it the gate, the zeroing and the emptying - then uploads over the loan range
   anyway. The cache's slots there had been refilled (the ledger from the *previous* chunk), so they were resident
   with `expert = Some` when the upload overwrote them. That is the NaN, and it explains all three of its shapes: it
   needs a **second** chunk (turn 2 onwards), it needs the refill to have repopulated the range first (a turn or two
   later), and `BORROW_STILL` never fired because the check never ran.

### 17.2 The fix

The once-per-chunk work moved out of `borrow_buffers` into `begin_chunk`, called from `forward` at layer 0 - the same
place the chunk's identity is decided - so it runs **whether or not the chunk had to allocate**: gate, order the copy
stream after the cache's work, zero the span, empty every slot the loan covers, settle the queued copies. Plus a
counter in `empty_range` for the other silent path (`BORROW_SKIPPED`: a resident slot with no storage, which cannot be
tested for overlap and used to be skipped without a word).

### 17.3 Measured after the fix

| measurement | before | after |
|---|---|---|
| chat, loan on, 40 measured turns | **4 failures in 96 turns, every run** | **0 failures**, 0 `internal error`, 0 `BORROW_STILL` |
| chat, loan on | - | TTFT >= 2 p50 997 ms, p90 **1 946 ms**, TPOT 42 ms, e2e 36.3 s |
| chat, loan off (shipped) | TTFT >= 2 1 083, p90 2 264, TPOT 43, e2e 36.4 s | - |
| 1 216-token prompt x 5 on one load, loan on | NaN | 33.86 ms TPOT, no NaN |
| cache ladder with the loan | 3.2 GB unusable | **2.2 vs 3.2 GB = 33.50 vs 32.36 ms**, hit rate 66.0 -> 72.4%, PCIe 14.9 -> 13.1 GiB a round |

Gates: oracle `frac 0` + `--no-lookahead` **byte-identical**; 56 tests.

The loan is therefore **sound** on this workload, and it is also why `--expert-cache-gb 3.2` now runs a long prompt.
It still stays **off by default**, for the reason §16 established and this section repeats: the ladder is flat, so a
bigger cache is not worth what the loan charges (one wipe of ~714 slots and its refill, 2.4 s per conversation at
2.2 GB). The mechanism is kept because it is the only way to run a cache larger than the free VRAM allows, and its
cost is now known to be bounded by the refill rather than by correctness.

---

## 18. Where a prompt chunk's time actually goes: 75% PCIe at the link rate, 25% dense work, and overlap gets half of it (2026-11-03)

§11.2 and §14.3 both pointed at "half of a chunk is per-layer latency rather than transfer". That was a guess made
from a division (23.4 GiB / 13.4 GB/s = 1.87 s against a 3.6 s wall). With the loan repaired it can be measured, and
it was wrong: the transfers are already at the link rate, and the "per-layer latency" is the model's own dense work.

### 18.1 The instrument

`NAIWA_STREAM_DBG=1` already printed, per layer, `dma_wait`, `compute` and the upload's queue time. It now also
prints **`gap`**: the time from one layer's entry into `forward` to the next layer's, which is a whole layer - our
expert path *plus* the model's attention/mixer/PLE work for it. Summing the 48 gaps of a chunk gives its host
timeline, so `sum(gap) - sum(compute)` is exactly what the model does outside the expert path.

Two runs, one long prompt (1 216 tokens, one chunk of 1 216 rows), shipped 2.2 GB cache:

| | chunk, entry-to-entry | of which our expert path | of which model-side | per-layer gap p50 |
|---|---:|---:|---:|---:|
| stride 1 (shipped, no loan) | **2.55 s** | 0.14 s | **0.62 s** | 55.4 ms |
| stride 2 (loan, two ranges) | **2.29 s** | 0.17 s | **0.36 s** | 48.8 ms |

### 18.2 The transfers are not the problem, and the 10 GB/s was a mirage

With `NAIWA_STREAM_SYNC=1` (blocking uploads, so the copy's own wall time is what is reported) every layer moves at
**13.0-13.4 GB/s**, i.e. the link's ceiling: 96 transfers, 47.9 GiB, 3.86 s, **12.1 GB/s aggregate** including the
first two slower ones (10.9, 10.5 - cold cudaMalloc and page-table work). The async path reported 10.0-12.8 GB/s
because `copy` there is the *queue* latency, not the transfer.

So the floor is real: **23.4 GiB at 13.4 GB/s = 1.87 s**, and a chunk of 1 216 rows takes 2.55 s (stride 1) or 2.29 s
(stride 2) - **1.36x and 1.22x the link floor**, not 2x.

### 18.3 What that means for the two levers people reach for

- **Overlap (the double buffer) is worth ~0.27 s a chunk (-10.5% of its wall, -270 ms)**, which is what it can hide
  of the 0.62 s the model spends outside the expert path. It cannot do better than that on this card, because the
  copy is 75% of the layer and there is only ~13 ms of compute a layer to hide behind ~40 ms of transfer. So M18's
  19% was never available here, and M33's -4.3% was measured under the broken loan (private buffers, cache emptied);
  with the loan sound and the alias real it is **-10.5%**.
- **Chunking bigger does not help either**: every chunk streams every layer once, so the traffic per token is
  23.4 GiB / chunk, and the wall is that traffic at the link rate plus a fixed ~13 ms a layer of dense work. Only
  fewer bytes per token helps, and the ways to get them are the expert cache (too small - §16 measured the ladder
  flat) or a smaller quant.

### 18.4 What is left in the prompt path

1. **The ~0.36 s a chunk the double buffer still leaves on the table** (0.62 - 0.27). It is the pipeline's fill and
   drain: layer 0's upload cannot overlap anything and the last layer's compute cannot hide behind a copy. That is a
   few percent and it needs a different mechanism (a ring, i.e. more than two buffers), which §9 already deferred.
2. **The first two transfers (10.5-10.9 GB/s against 13.4)** - cold-allocation effects: `NAIWA_SHAPE_TRIM` shows the
   pool reuse that follows costs ~31 us, so this is cudaMalloc's first-touch, not the link. Worth ~30 ms a chunk.
3. Everything else is the link. On this card, the prompt path is a copy job.

### 18.5 So should the loan be on by default? Not on this evidence

Two more chat runs after the NaN fix, 3 rounds and 24 measured turns each, 0 failures in both:

| arm | TTFT t1 | TTFT >= 2 p50 | p90 | TPOT p50 | e2e per conversation |
|---|---:|---:|---:|---:|---:|
| loan on | 1 031 ms | 1 062 ms | 2 280 ms | 44 ms | 33.5 s |
| loan off | 976 ms | 971 ms | 2 326 ms | 44 ms | 36.2 s |

against the earlier 5-round run with the loan on (TTFT >= 2 997 ms, p90 1 946, TPOT 42, e2e 36.3). The three arms
overlap: on the `short` corpus the loan is **neutral** - which is what the corpus predicts, since its first turn is
40 prompt tokens and the loan only pays on a prompt long enough to be a chunk worth overlapping.

So the default stays **off**, and the reason is now precise: on the workload the harness measures, the loan is a wash;
its measured value is on a long single prompt (**-270 ms a chunk, -10.5%**) and on being the only way to run a cache
larger than free VRAM. The measurement that would decide it is a `long`/`doc` class A/B, which the corpus has: `doc`
is 7 turns and `long` 3, both of which grow into real chunks.

### 18.6 Is the copy at the link's ceiling? Yes - and the link is Gen3 x16, not the 13.4 GB/s the notes assumed

`nvidia-smi` reads `pcie.link.gen.current = 1` at rest (the card downclocks an idle link); sampling it **during** a
load shows **Gen3, x16** (with Gen2 in between), which is the slot's maximum (`link.gen.max = 3`, `width.max = 16`).

| | |
|---|---|
| Gen3 x16, one direction, raw | 8 GT/s x 16 = 15.75 GB/s |
| practical H2D on this link | ~13-14 GB/s |
| **measured, blocking uploads** | **13.0-13.4 GB/s** a layer; 50.3 GB in 3.86 s = **13.0 GB/s** aggregate |
| measured, and the ideal for the same bytes | 1.93 s a chunk against 1.90 s at 13.2 GB/s |

So the copy runs at **~83% of the raw link and at the top of what a Gen3 x16 slot delivers in practice** (the 17% is
128b/130b encoding plus TLP overhead, not this code). The 13.4 GB/s in the earlier notes was right by accident - it is
the *practical* rate, not the theoretical one - and the conclusion it supported is unaffected: a chunk's transfers
cannot be made faster on this machine.

The only copy-side slack left is the first two layers of a chunk (10.5 and 10.9 GB/s against 13.2 steady, ~30 ms a
chunk, cold first-touch of the slab), and the copy stream's idle time while the model computes (0.62 s a chunk at
stride 1, 0.36 s at stride 2) - which is the overlap question of §18.3, not a bandwidth question.

### 18.7 Per layer, transfer against everything else

One chunk of 1 216 rows, blocking uploads (`NAIWA_STREAM_SYNC=1`, so the copy's own wall time is what is reported),
stride 1, grouped by the layer's weight group (the model has seven, one per quant mix):

| layer bytes | layers | copy p50 | copy rate | layer total p50 | "everything else" p50 |
|---:|---:|---:|---:|---:|---:|
| 368 MiB | 5 | 29.7 ms | 13.0 GB/s | 55.6 ms | 25.5 ms |
| 418 MiB | 3 | 33.8 ms | 13.0 GB/s | 52.6 ms | 18.8 ms |
| 456 MiB | 1 | 36.7 ms | 13.0 GB/s | 46.8 ms | 10.1 ms |
| 481 MiB | 15 | 38.7 ms | 13.0 GB/s | 54.9 ms | 16.4 ms |
| 531 MiB | 14 | 42.8 ms | 13.0 GB/s | 53.8 ms | 10.5 ms |
| 568 MiB | 9 | 45.7 ms | 13.1 GB/s | 55.2 ms | 9.3 ms |
| 650 MiB | 1 | 52.3 ms | 13.0 GB/s | 59.1 ms | 6.8 ms |
| **all 48** | 48 | **40.4 ms** | **13.0 GB/s** | **~55 ms** | **13.9 ms** |

Per chunk that is **1 940 ms of copying and 615 ms of everything else** (2 555 ms), i.e. 76% / 24%. Over 1 216 rows
that is **1.6 ms of transfer and 0.5 ms of compute a prompt token**.

Two things the table says that a single average hides. A layer costs ~55 ms whatever its size, because the copy and
the rest trade places: the 368 MiB group moves 40% fewer bytes than the 650 MiB one and spends 3.7x more time outside
the copy. And the copy rate is flat at 13.0 GB/s across all seven groups, i.e. the transfer is at the link for every
shape - the per-layer differences are bytes, not efficiency.

What overlap buys, measured the same way at stride 2: the chunk's "everything else" drops from 615 to ~360 ms, so a
second buffer hides about 40% of the non-copy time; the copy's 1 940 ms is untouched, because it is the link.

### 18.8 The prefetch exists, it is off by default, and with it the link is still idle 22%

`forward` has prefetched the next layer since the two-buffer design landed:

```rust
if stride == 2 { self.load(layer + 1, stride)?; }   // queued before this layer's compute
```

and `NAIWA_STREAM_DBG` now reports the host time that costs (`pf`): **p50 66 us, max 86 us**, so it is a real async
prefetch, not a disguised wait. What decides whether it runs is `stride == 2`, i.e. whether a second 650 MiB layer
buffer exists. In the shipped configuration it does not - it is exactly what the loan buys - so **the shipped prompt
path has no overlap at all**: per layer, `[copy 40 ms] -> [compute 14 ms]`, and the chunk is the sum of both.

With the loan (stride 2), on the same 1 216-row chunk:

| | Σ over 48 layers | effective rate | link idle |
|---|---:|---:|---:|
| stride 1 (shipped) | 2 555 ms | 10.1 GB/s | 25% |
| stride 2 (loan, prefetching) | **2 322 ms** | 11.1 GB/s | **22%** |
| the same 23 961 MiB at 13.0 GB/s | **1 800 ms** | 13.0 GB/s | 0% |

So the prefetch is worth 233 ms a chunk, and **0.5 s is still on the table** - but not from more prefetch depth in
the sense of "start earlier". One layer of lookahead is exactly one copy's worth of buffer, so the copy stream runs
dry the moment the host is late by a few milliseconds (the per-layer host work between one `forward` call and the
next is ~8 ms, and the link idles for it). Keeping the link busy needs **more than two buffers** - a ring - which is
the item §9 deferred, and the number above is what it is worth: **up to 0.5 s of a 2.3 s chunk, ~16%**.

The counterpart in the other direction: overlap cannot shorten the copy itself, so no arrangement of buffers gets a
chunk below its 1.80 s of transfer. Everything else in the prompt path is the model's own dense work (615 ms at
stride 1, ~360 ms at stride 2).

---

## 19. Prefill against Strata, with overlap on: 1.31-1.37x behind, and the batch size is not the reason (2026-11-03)

Same GGUF pack, same card, same context (4 096), both **cold** (`cached_tokens = 0` on every measured request; strata's
own accounting prints `0 reused + N read`), three unique random prompts a size, median of three:

| prompt | naiwa (loan on, stride 2, 2.5 GB cache) | strata (`--prefill auto`, 1.98 GiB cache) | strata's own |
|---|---:|---:|---:|
| 1 072 tok | TTFT **2 917 ms** -> 367 tok/s | TTFT **2 224 ms** -> 482 tok/s | 471-497 tok/s |
| 3 404 tok | TTFT **4 729 ms** -> 720 tok/s | TTFT **3 447 ms** -> 988 tok/s | 984-1003 tok/s |

So strata reads a prompt **1.31x** (1 k) to **1.37x** (3.4 k) faster. Splitting each arm by its own two terms
(TTFT = fixed + per-row, from the two sizes):

| | fixed part | per extra row |
|---|---:|---:|
| naiwa | 2 084 ms | 0.78 ms |
| strata | 1 662 ms | 0.52 ms |

### 19.1 The batch size: strata's is *smaller* here, not larger

Strata's chunk is chosen by `--prefill auto` as the largest 256-aligned chunk whose *loan* fits, capped by
`prefill_auto_max = 8192`, the context, and the ring that competes for the same budget (`generate.cpp:5760-5800`).
On this card its log says:

```
strata serve: prompt chunk auto: 2816 tokens, a 199-slot ring
strata serve: the prompt path borrows 874 CUDA0 cache slots (1.68 GiB)
```

**2 816 tokens**, not 8 192 - the 8 192 is the ceiling, and the 90%-of-slots loan cap plus the ring bring it down. The
chunk ladder it falls back through is `{6144, 4096, 3072, 2048, 1536, 1024, 512}`, and below
`stream_all_min = 1024` tokens it stops streaming every non-resident expert and streams only the routed ones (with the
CPU share taking some).

Naiwa's chunk on the same card is **4 608** (planned 5 376 at warmup, re-decided to 4 608 at the first chunk), so for
the 3 404-token prompt **naiwa runs one chunk and strata runs two** (2 816 + 588) - strata moves *more* bytes and is
still faster. Batch size is therefore not what this gap is made of.

### 19.2 What it is made of

- **The fixed part (2 084 against 1 662 ms, +422 ms)**: both engines stream the experts they do not hold, so this is
  the copy. Naiwa's copy of a chunk is 1 940 ms at 13.0 GB/s (§18.7) - i.e. it is already at the link - and strata's
  1 662 ms corresponds to ~21.5 GiB at the same rate. Strata hides more of its copy (its dense work runs behind it);
  naiwa's prefetch, one layer deep, cannot (§18.8).
- **The per-row part (0.78 against 0.52 ms a row, +50%)**: this is the model's own work - attention over the growing
  prefix, the per-layer embedding gather, the MoE GEMM for the chunk's rows - in *our* forward path. At 3 400 rows it
  costs naiwa ~2.7 s against strata's ~1.8 s. Nothing about the link or the cache enters here.
- A structural difference worth naming: **naiwa's prompt path ignores the expert cache entirely** - a chunk uploads
  every layer's full stack (23.4 GiB per chunk) whether or not those experts are resident. Strata streams only the
  non-resident ones, so its 1.98 GiB cache takes ~8% off its prompt traffic. Ours takes nothing off it.

### 19.3 So the next prompt-path lever is the per-row work, not the transport

`0.26 ms a row` x 3 400 rows is ~0.9 s - the larger half of the gap, and it is ordinary model work (attention, PLE
gather, the chunk's MoE GEMM) rather than streaming. That is where a prompt-path milestone should start, with the
fixed part's remaining 0.3 s (a ring, §18.8) second.

---

## 20. Where the per-row work is: a kernel profile, and a 0.45 s stall the profile found (2026-11-03)

§19 left the gap at "0.78 ms a row against strata's 0.52". This is the profile of one cold 3 404-token prompt
(`nsys`, `harness/profile`), split by kernel family, next to the same for a 1 072-token one:

| kernel family | 1 072 rows | 3 404 rows | slope |
|---|---:|---:|---:|
| quantized GEMM (`mul_mat_q`, all dtypes, + `quantize_mmq`) | 0.352 s | **0.862 s** | +0.22 ms/row |
| attention (`q4_qsa_attention_kernel`) | 0.001 s | **0.525 s** | +0.22 ms/row |
| GDN recurrence (`gated_delta_rule_*`, `gdn_*`) | 0.092 s | 0.294 s | +0.09 ms/row |
| **CPU-expert doorbell wait** | **0.486 s** | 0.214 s | fixed |
| bf16 GEMM | 0.050 s | 0.116 s | |
| hyper-connection mix/combine | 0.028 s | 0.092 s | |
| MoE route/reduce | 0.071 s | 0.075 s | |
| everything else (cast/copy/other) | 0.122 s | 0.164 s | |
| **kernel total** | **1.20 s** | **2.34 s** | |
| wall (first token) | 3.19 s | 4.63 s | |

Three things to read out of it.

**1. `doorbell_wait_copy_f32_kernel` is a stall, not work.** It is this crate's own handshake - the device spins on a
mapped counter until the *CPU* has published a CPU-computed expert's result (`src/doorbell/`). During a prompt chunk
the GPU has every expert it needs (the chunk uploaded the layer), so nothing should be waiting on the CPU at all. It
is 40% of the kernel time at 1 k rows and 9% at 3.4 k, and it is **fixed per step**, which is why it is invisible at
long prompts and dominant at short ones. The cause is the lookahead/prediction machinery still running per layer
during a prompt step.

**2. Killing it is worth 0.45 s.** `NAIWA_PREFILL_QUIESCE=1` (no prediction work during a chunk, and the previous
step's queued jobs dropped) on the same prompts:

| prompt | default | quiesced |
|---|---:|---:|
| 1 072 tok | 2 917 ms (367 tok/s) | 2 911 ms (368) |
| 3 404 tok | 4 729 ms (720 tok/s) | **4 280 ms (795 tok/s)** |

-9.5% at 3.4 k, nothing at 1 k (the stalled work is fixed, so it hides at larger sizes). Against strata's 3 447 ms
that moves the gap from 1.37x to **1.24x**.

**3. The remaining per-row cost is attention and the quantized GEMMs, in equal parts (0.22 ms a row each).** The
attention term is the one that will not age well: `q4_qsa_attention_kernel` is ~nothing at 1 k rows and 0.525 s at
3.4 k, i.e. it is strongly superlinear (the compressed/indexed attention path), so at longer contexts it becomes the
whole prompt.

### 20.1 What the profile says to do next

- **Make the chunk quiesce the default** (pending the chat A/B: it must not cost decode hit rate after a prompt).
- **Attention first, then GEMM**: 0.22 ms a row each today, and attention is the one that grows superlinearly. That is
  the milestone after this one - it is a kernel question in the fork, not a scheduling question here.
- The copy is not on this list at all: it is the 2.3 s of wall that no kernel occupies, and §18.8 already has the only
  remaining answer for it (a ring behind the two buffers).

### 20.2 The chunk quiesce is the default now, and what that leaves

`forward` quiesces the prediction machinery at layer 0 of a chunk (`NAIWA_PREFILL_NOQUIESCE=1` for the A/B arm), and
the old `NAIWA_PREFILL_QUIESCE` block in `borrow_buffers` is gone with it. Verified with the default in place:

| prompt | before (§19) | now |
|---|---:|---:|
| 1 072 tok | 2 917 ms -> 367 tok/s | 2 913 ms -> 368 tok/s |
| 3 404 tok | 4 729 ms -> 720 tok/s | **4 313 ms -> 789 tok/s** |

Gates: oracle byte-exact, 56 tests, and the chat harness is the better arm with it (p90 2 124 against 2 347 ms, e2e
29.9 against 37.1 s, 0 failures in 48 measured turns across the two arms).

With that in, the standing comparison against strata on the same cold prompts is **1.24x at 3.4 k tokens** (4 313
against 3 447 ms) and **1.31x at 1 k** (2 913 against 2 224), and the profile says where the rest is: attention and the
quantized GEMMs at ~0.22 ms a row each, with attention the superlinear one.

### 20.3 Is dequantization in the prompt path? No - and the attention kernel's problem is that it has no tensor cores

§20 put attention and the quantized GEMMs at ~0.22 ms a row each. Asked directly: does either of them
**dequantize weights** first? The profile answers both questions, and the answer changes where the work is.

**Dequantization: absent from the prompt path.**

- Every one of the model's nine quantized types runs ggml's **MMQ** kernel - `mul_mat_q<(ggml_type)12/13/14/18/20/21/22/23/42>` -
  which dequantizes *inside* the dot product against a q8_1-quantized activation (`quantize_mmq_q8_1`, 47 ms) and never
  materializes a dequantized block. The profile contains **no `dequantize_*` kernel at all**.
- The only conversion kernels are activation-sized: `cast_f32_bf16` 29.9 ms (1.3%, 695 launches), `cast_bf16_f32`
  5.6 ms, plus elementwise `ucopy_bf16`/`bmul_bf16`/`badd_bf16` (21/18/8 ms). Nothing weight-sized.
- One kernel really does dequantize: `mmvq_gguf_q6_k_bf16_plain_cuda1`, 9.2 ms, 205 launches - a Q6_K-to-bf16 matvec
  for the one-row case, i.e. **decode**, not the prompt.
- The upload path does not dequantize either: a chunk copies the stacks' raw quantized bytes (386 MiB for a 1 216-row
  chunk at 13 GB/s, §13), which is why the transfer is that small in the first place.
- `q4_qsa_attention_kernel`'s "q4" is the **model family (Qwen4)**, not 4-bit: its K/V are `const T *__restrict__` in
  bf16 (`qwen4_exp.cu:777`). The KV cache is not quantized, so there is nothing to dequantize.

**Attention's actual limit: it is a scalar-FMA kernel.** From `qwen4_exp.cu`:

- head dim is fixed at 256 and **each lane owns 8 dims** (`qv[hh][8]`, `acc[hh][8]`, `q4_load8`), 4 warps a block, up to
  4 q heads a warp; grid is `(tokens, kv_heads=2, splits)`. K/V are loaded per warp and reused across that warp's q
  heads, and one kv head's selection (3.5 MiB) lives in L2 - so this is **not** a DRAM-bandwidth problem.
- The arithmetic is plain `float` FFMA: for a 3 404-token prompt, 12 QSA layers x (3 404^2 / 2) pairs x 24 heads x 256
  dims x 2 = **~854 GFLOP in 0.525 s = 1.6 TFLOP/s**, against ~29 TFLOP/s of bf16 tensor-core peak on this card.
  **~5% of peak** - the kernel never touches an MMA pipe.
- That is also why the cost is superlinear (1 k rows ~0, 3.4 k rows 0.525 s): the pairs grow with the square of the
  prefix while the throughput per pair stays at a scalar-FMA rate.

So the next prompt-path milestone is not "avoid dequantization" (there is none) and not the copy (at the link): it is
**moving `q4_qsa_attention_kernel` onto tensor cores**, which is worth ~0.4-0.5 s of a 3.4 k prompt (~10%) today and
grows quadratically with context. Second is the MMQ aggregate (0.86 s), which is already the efficient path.

### 20.4 The code-level comparison with strata on this exact part: they run the same math on tensor cores

`qsa_prompt_attn.cu` (1 525 lines) against our `qwen4_exp.cu:777`, for the same model and the same selection scheme:

| | strata | naiwa |
|---|---|---|
| selection | indexer top-k + compress ratio, per-token item list, split-K partials + reduce | **same scheme** (`selected`/`n_selected`, `ratio`, `part_acc`/`part_ml`, reduce kernel) |
| block shape | 4 warps (128 threads), `HD = 256`, `G = 12` q heads a kv head, 32-cell chunks | 4 warps, `HD = 256`, up to 4 q heads a warp, 4-item inner loop, **8 dims a lane** |
| math | **`mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`** (and m16n8k8) + `cp.async` (sm80+) | `q4_load8` + **`float` FFMA** |
| KV | **quantized**: mode 0 fp16, mode 1 int8 codes + a fp16 scale a 64, mode 3 K8V4, mode 4 q4_0 + a scale a 32. The codes are made into *exact halves* (mantissa tricks, "1024 + (c + 128) ... minus 1152") and fed to the same fp16 MMA, scales applied in FP32 per group | bf16 only (`const T *k_cache`) |
| fallback | keeps a scalar kernel (`native_flash_attn.cu`, **0** MMA intrinsics, from llama.cpp's fattn-vec) for pre-sm_75 and AMD | - |

Two things follow, and they are both about code, not about scheduling.

1. **Ours is their *fallback*.** Strata went through the scalar kernel, kept it for old cards, and moved the prompt
   path to `mma.sync`; the fork has **no `mma.sync` anywhere in the attention path** (`qwen4_exp.cu` has none, only
   `moe_gemm_wmma.cu` for bf16 experts, which our quantized experts do not use). That is the 1.6 TFLOP/s of §20.3
   against ~29 TFLOP/s of fp16 tensor-core peak - and it is a kernel to write, not a flag to flip.
2. **They quantize what we do not.** Their int8/q4 KV halves the bytes a pair costs *and* rides the same fp16 MMA by
   packing codes as exact halves; our KV is bf16 and every pair pays 2 bytes and four FFMA per 8 dims.

The size of the prize, against the measured gap: at 3 404 tokens the two engines differ by 866 ms (4 313 against
3 447) and attention is 525 ms of our side. Moving it onto tensor cores at a conservative 25-40% of peak would take it
to 0.05-0.11 s, i.e. **0.4-0.45 s, about half of the remaining gap** - and the share grows quadratically with context.

### 20.5 Attempting attention: the GPU numbers are stable, the wall clock is not, and the kernel's real defect is a shuffle chain

Before rewriting anything, two things had to be established. Both were.

**1. The kernel profile is stable; the wall clock is not.** The same build and prompt, profiled twice hours apart:

| | first | second |
|---|---:|---:|
| `q4_qsa_attention_kernel` | 524.5 ms | 526.2 ms |
| `mul_mat_q` (all types) | 861.8 ms | 862.6 ms |
| `gated_delta_rule_recurrence` | 237.9 ms | 239.9 ms |
| kernel total | 2.34 s | **2.20 s** (the doorbell stall of §20 is gone) |

while TTFT for the same configuration moved from **4 313 ms to 6 303 ms** and repeated requests degrade within a run
(4 963 -> 6 422 -> 6 303 ms). The cause is host memory, not the GPU: this model is 54 GiB of weights (23.4 experts +
26.8 PLE) in 60 GiB of RAM, swap is in use (1.5 GB), and the prompt path reads the expert stacks from host memory. The
GPU-side profile sampled a 4.00 s prompt in the same session. **So: kernel numbers are comparable across sessions,
wall clocks are only comparable within one run, and the box needs the page cache left alone (or more RAM) before any
prompt-path wall-clock claim.**

**2. The cheap knob is not the lever.** `Q4_ATTN_MAX_HEADS_PER_WARP` was 4 for a GQA group of exactly 12 q heads, so
a quarter of the head loop is skipped by its guard; setting it to 3 covers 12 exactly. Measured: slower in the one
clean run (5 607 against 4 313 ms), and with the host thrashing the comparison cannot be trusted either way.
**Reverted** - the fork is unchanged.

**What the code says the defect is.** The inner loop computes each score with a lane-local 8-dim dot and then
**reduces it across all 32 lanes with a five-step `__shfl_xor` chain, for every item batch**:

```cpp
for (int o = 16; o > 0; o >>= 1) for (int j = 0; j < 4; ++j) dot[j] += __shfl_xor_sync(0xffffffff, dot[j], o);
```

That is 20 shuffles (five dependent steps, ~25 cycles each) per 4 items per q head, against 32 FMAs of actual work - so
the kernel spends its time in shuffle latency, not arithmetic, which is why 854 GFLOP lands at 1.6 TFLOP/s, ~7% of
even the *scalar* FP32 peak and ~5% of bf16 tensor peak. Batching more items to amortise it is not available either:
`kf`/`vf` are already 64 registers at 4 items, and 8 items would spill.

Strata does not have this problem because `mma.sync` produces the whole 16x8 score tile in one instruction and the
softmax needs a 2-level (4-lane) reduction instead of a 5-level (32-lane) one. That is the rewrite, and the design is:

- `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`, m = 16 q-head rows (12 real), n = 8 items, k = 16 dims
  (16 k-steps for head_dim 256); the indexer selection and split-K partials stay exactly as they are.
- softmax per row over the 8 columns held by 4 lanes: `__shfl_xor` with masks 1 and 2 only.
- PV as m = 16 heads, n = 8 dims (32 tiles for head_dim 256), k = 16 items, with K/V staged through shared memory
  (`cp.async`) so the fragments are loaded once per block instead of once per warp.
