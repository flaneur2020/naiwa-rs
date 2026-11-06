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
| loan, one r