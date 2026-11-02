# M33 - the prompt path borrows the expert cache (strata's loan)

**Status:** open, **nothing built**. Feasibility recorded 2026-11-02 (no code changed). Numbering note: M29-M32 are
withdrawn (`roadmap.md`, "M29-M32 aborted 2026-11-01"), so this milestone takes the next free number.
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
