# M19 — one VRAM arena: the prefill stream borrows the expert cache

Status: **not started** (opened 2026-10-21).
Parent: [m18-prefill-transport.md](m18-prefill-transport.md).
One mechanism this milestone: **a single expert-VRAM arena, partitioned once at startup, whose prefill region is
the expert cache when no prompt is being prefilled.**

## Why, in one measurement

M18 ended on a trade we could not win: buying the prefill stream's second buffer costs 450 MiB of expert cache
(turn-1 4 431 -> 3 654 ms, -18%) and the smaller cache costs more on decode (TPOT 44.9 -> 51.0 ms with the
acceptance rate unchanged, ~4 s lost per conversation against 0.8 s saved). The two uses need the *same* VRAM
at *different times*, and today they are separate reservations:

| consumer | today | when it is live |
|---|---:|---|
| expert cache, `--expert-cache-gb 2.3` | 2 355 MiB (1 208 slots, 25/layer of 256) | decode |
| prefill stream, chunk 8192, 1 buffer (`free` funds it) | 1 749 MiB reservation (650 buffer + 522 activations + 65 slack + 512 headroom) | prefill |
| prefill stream, 2 buffers | 2 399 MiB reservation | prefill |
| staging ring (16 slots) | ~33 MiB | decode |
| drafter pool (MTP head, Q2_0) | 675 MiB | both |
| recurrent state, `--max-seqs 4` | 459 MiB | both |

Prefill and decode alternate, so **2.4 GiB of buffer is idle during decode and 2.3 GiB of cache is idle during
prefill**. Sharing them would give decode 1 208 -> ~2 200 slots (+83%) *and* keep two prefill buffers, instead
of choosing.

## The static shortcut, measured and refuted (2026-10-21)

Before the arena: can the second buffer simply be *reserved* from free VRAM? The plan's formula says it fits if
the 512 MiB slack floor drops. It does not - `NAIWA_PREFILL_SLACK_MIB` was added to test exactly that:

| config | plan says | free | outcome |
|---|---:|---:|---|
| chunk 3072, slack 192 MiB, 2 buffers | 1 894 MiB needed | 1 983 MiB (**89 MiB margin**) | **`CUDA_ERROR_OUT_OF_MEMORY`** |
| chunk 2816, slack 192 MiB, 2 buffers | 1 888 MiB needed | 1 985 MiB (**97 MiB margin**) | **`CUDA_ERROR_OUT_OF_MEMORY`** |

So the formula's `activations + activations/8` under-counts the peak by **more than 89 MiB** at these chunks (the
paged-attention scratch and the step's own buffers are not in it), and the true slack is above the 231 MiB the
arithmetic would allow (`1983 - 1300 - 402 - 50`). Freeing the only other lever - `--max-seqs 4 -> 2` returns
229 MiB - lands on a knife edge with no margin, and `--max-seqs 1` is a concurrency decision.

**Therefore the second buffer cannot be funded from free VRAM at all.** It has to be memory the cache already
owns, which is strata's `borrow` exactly ("the top slots of the expert cache"). That is the milestone, and the
phases below are ordered around it rather than around a static reservation.

## Single-user (decided 2026-10-21): what it simplifies, and what it does not

The engine only has to serve one user, so concurrency needs no consideration. That removes the hardest part of
the loan design - there is no decode step to protect while another request prefills, and the loan can be held
for the whole prompt phase (as strata does) instead of being released per chunk or per admission epoch.

It does **not** hand us the VRAM, which was worth checking: `--max-seqs` looked like the obvious lever (M10
measured 112 MiB per sequence slot at runtime), but the recurrent-state pool is allocated **lazily**, so it is
not a plan-time budget:

| `--max-seqs` | free VRAM at the warmup plan |
|---|---:|
| 1 | 2 338 MiB |
| 4 | 2 335 MiB |
| 16 | (same shape; the pool grows on the first decode, not at load) |

and the first chunk sees 1 983-1 986 MiB in all cases. `--max-seqs 1` also changes nothing about the buffer
decision: chunk 3072 needs `1 300 + 402 + 50 + 512 = 2 264 MiB` and the plan still granted one buffer.

So single-user is a *simplification* of the loan, not a substitute for it. The borrow stays the plan.

## The allocator: one pool, two watermarks (decided 2026-10-21)

The design of record is not "carve a region at startup" but a **two-sided allocator** over one pool of uniformly
sized slots:

```
 slot 0                          prefill_high      gap      cache_low                       total
 |<------ the stream's run ------->|<--- free --->|<------- the expert cache --------------->|
```

- **The cache fills from the top down** (`cache_fill`), so its hottest resident sits nearest `total`.
- **The prefill stream grows from the bottom up** (`borrow`), in whole-layer runs.
- **When the stream needs more than the gap, the cache gives up its lowest slots first** (`borrow().evict`),
  which are exactly its coldest - the user's rule, and it means the eviction order *is* the allocation order, so
  no second policy (no LRU scan, no separate victim selection) is needed.
- **`release`** returns the run; those slots are refillable by the cache immediately, and nothing is saved or
  restored because cache contents are copies, not dirty state.

The rate instrument from M18 (`cargo test h2d_ -- --ignored --nocapture --test-threads=1`) still reports
**13.4 GB/s** for every pinned case when run one at a time; running them in parallel makes the threads contend
for the link (7.4-8.9 GB/s), which is why they are `#[ignore]`d out of the normal suite.

Implemented as `src/moe/arena.rs` (`SlotArena`, no GPU, no allocation - it only hands out slot indices and the
eviction list, which keeps the policy testable), with six unit tests pinning the four rules above: the cache
fills top-down, the stream only ever grows its run, eviction is coldest-first, release makes the run refillable,
and the watermarks never cross. `6/6` passing.

### The API constraint found while integrating (2026-10-21)

`QTensor::new(storage, shape)` takes a whole `QStorage`, and candle exposes **no way to build a `QTensor` over
a sub-range of a `QCudaStorage`** (no `slice`/`narrow`/offset constructor on the CUDA quantized storage). So
"one arena, and every layer gets a `QTensor` view into it" is not expressible: a `[total, N, K]` bank can only
be a `QTensor` for one shape.

Two consequences for the plan:

- the arena has to be built **per shape group** (`[total_group, n, k]`, one allocation per group; this model has
  two groups), and a layer's range inside its group's arena is addressed by `SlotMemory` views - which *are*
  offset-able (`ptr + i * stride`) and are what the fill path already uses;
- a layer's compute path must therefore keep building its per-expert/hit views from `SlotMemory`, not from the
  group's `QTensor` - which is what `expert_qtensor_on_device` already does, so this is a check rather than a
  change.

That is the remaining unknown for step 1 below: whether every consumer of a layer's `Bank` can work from
`SlotMemory` alone. It is checkable by inspection before any code moves, and it is the first thing to do next.

### What integration still needs (P1's remaining work)

1. **One uniform pool**: replace the per-layer `alloc_bank` allocations with a single allocation of
   `total x slot_bytes(max shape)` (strata's `ExpertCache::open(..., blob_bytes)` shape), so a slot is a slot for
   any layer and a contiguous run can be a layer buffer.
2. **Slot ids instead of `Bank.first + index`**: the cache's per-layer bookkeeping becomes a list of slot ids
   from `cache_fill`, because the arena's evictions take arbitrary (lowest) slots rather than a per-layer tail.
3. **Prefill borrows**: `plan` turns its footprint into `ceil(footprint / slot_bytes)` slots, calls `borrow`,
   invalidates `evict`, and aliases the run as `[E, N, K]` for the layer being prefilled; `release` after the
   prompt.
4. **Refill**: after `release`, the cache refills the run from the top down as it would after any eviction
   (~333 slots x 1.95 MiB per layer buffer, ~97 ms of DMA at 13.4 GB/s).

## The layout blocks the loan (analysis, 2026-10-21)

Working through P1 in the code found the real obstacle, and it is not the plumbing - it is how the cache is laid
out:

| fact | number |
|---|---:|
| one prefill buffer (a layer's whole expert set) | 650 MiB = **333 cache slots** |
| one cache bank (one layer, `--expert-cache-gb 2.3`) | 25 slots |
| banks per model | 48, one allocation each, per-layer shape |

`CopyStream::alloc_bank(stack, slots)` allocates **one allocation per layer**, shaped `[slots, N, K]` for *that*
layer, and the stream's buffers come from `alloc_like(stack)` - an allocation shaped `[E, N, K]`. So a region
that could be lent back to the cache has to be *one contiguous same-shape range*:

- borrowing one layer's own bank gives 25 slots = 49 MiB, 13x too small for a layer buffer;
- allocating a "spare bank" of 333 slots (650 MiB) works as a buffer, but it can only ever serve cache slots
  for **the one layer whose shape it was built for** - about 1/48 of the decode misses. That is not the +83%
  slot win the milestone is after;
- taking the 333 slots out of the cache *permanently* is worse than the M18 trade (M18 gave up 231 slots for
  -18% turn-1 and measured a net loss).

So the loan needs a **shape-agnostic pool**: slots that any layer can use for residency, addressable as one
contiguous range when the stream wants a layer buffer. Two layouts can do it:

| layout | how | cost |
|---|---|---|
| **A. one pool + an index map** | one allocation sized `total_slots x max_slot_bytes`; a per-layer map from its expert ids to pool slot indices (the cache's `Bank.first`/`Slot` indirection already has this shape, it is just per layer today) | an index translation on every hit and every fill; slot bytes rounded up to the max shape |
| **B. one pool, uniform slot size** | same allocation, every layer's slot padded to the max shape's bytes | simpler (no map), but a 481 MiB layer's slots waste ~8% |

Layout A is the honest version of strata's "top slots of the expert cache" (their cache is a flat slot pool:
`ExpertCache::open(n_slots, n_layers, n_expert, blob_bytes)` - one `blob_bytes` for all layers, which is exactly
layout B, with their generic MMAQ kernels not caring which layer a slot came from).

**Their layout is B**, and that is worth noting: `blob_bytes` is a single number for the whole pool, so a slot
is a slot for any layer, and a contiguous range of them can be any layer's buffer. That is why `borrow` is a
one-line proposition for them and a refactor for us.

## What strata does, verbatim

```cpp
/// `borrow`/`borrow_bytes`: device memory to carve every buffer from (the top slots of the expert cache,
/// lent for the prompt and refilled after it); null = allocate normally.
bool Prefill::init(..., void* borrow = nullptr, uint64_t borrow_bytes = 0);

/// With borrowed buffers: lay them out again for chunks of `chunk` tokens (at most `init`'s) in `borrow` -
/// a request lends only the slots its prompt needs. The stream must be idle (between prompts).
bool Prefill::relayout(int64_t chunk, void* borrow, uint64_t borrow_bytes, std::string& err);

/// Device bytes `init` needs for a chunk of `chunk` tokens (what a borrowed region must hold).
static uint64_t Prefill::bytes_needed(const ModelGeometry&, const SessionState&, int64_t chunk);
```

Four things to take from it:

1. **Not just the layer buffer**: *every* prefill buffer - chunk activations, MoE scratch, the streamed expert
   ring - is carved from the borrowed region, and `bytes_needed(chunk)` prices it up front. Our equivalent is
   the whole right-hand side of `prefill_stream::plan` (activations, slack, headroom).
2. **The region is the top slots of the cache** - an offset inside one allocation, not a second allocation.
   Our M10 experiment lent *banks* (per-expert slots with their own fences) and re-allocated 1.1 GiB per turn;
   that cycle is what fragmented VRAM (free 1 887 -> 1 489 MiB) and latched the stream off. A region inside one
   arena cannot do that.
3. **`relayout` per request**, because a request lends only the slots its prompt needs: a 400-token prompt does
   not need a 8 192-token chunk's buffers. That is how the cache gets its slots back on short prompts.
4. **"lent for the prompt and refilled after it"**: the slots were resident, so they come back dirty and must be
   refilled. That is a real cost (~1.4 GiB of DMA, ~100 ms) and it is smaller than the 4 s the M18 trade lost.

## The design for us

### One arena, three regions

```
VRAM for experts (one budget, decided at startup)
├── cache region     : n_cache_slots  × slot_bytes(shape)   LRU + lookahead fill, decode residents
├── borrow region    : max over layers of one layer's expert bytes (531 MiB today)
│                      + activations(chunk) + slack
│                      prefilled from `free` at startup; used as prefill buffers; loanable as cache slots
└── staging region   : 16 slots × slot_bytes(shape)          --pcie-frac ring (already a sentinel bank)
```

- The arena is allocated **once** (`cudaMalloc`/pool) and every region is an offset into it, so no
  allocate/free cycle, no fragmentation, and the plan never has to ask "is there room for a buffer?" again.
- `prefill_stream::plan` stops taking `free` and takes the borrow region's size; it chooses only the chunk, and
  it can no longer *decline* (today a 3 MiB miss latched the stream off for a whole run - M10's `long` turn at
  13 tok/s).
- The drafter pool (675 MiB) is a fourth region of the same arena for the same reason: it is permanent, and
  keeping it out of `free` is what makes the accounting honest (M16 had to cut the cache 3 -> 2.3 GB by hand).

### Shape rules (the part that has to be right)

- A cache bank is per layer and per shape: `[slots, N, K]` with that layer's stride. Mixed-quant GGUFs have
  two shapes today (481 and 531 MiB per full layer).
- A prefill buffer must be aliased as `[E, N, K]` for `MoEExperts::from_qtensors`, and only for the shape it was
  sized for. So the borrow region is sized to the **largest** layer and re-aliased per layer shape (the
  existing `alloc_like` already re-allocates on a shape change; with an arena it re-aliases instead).
- Slot indices must not collide: the staging ring already uses `STAGING_RING_FIRST = 1 << 30` as a sentinel;
  the borrow region needs the same treatment (`BORROW_REGION_FIRST`), so the LRU, the fill and the hit path
  never address it as a cache slot.

### Three phases, each with its own gate

**Ordering correction (2026-10-21).** An earlier sketch had P1 take the region out of the cache *permanently*
(`--expert-cache-gb` unchanged, cache slots reduced by the region) and P3 lend it back later. That ordering is
wrong: taking it permanently is exactly the M18 trade in a different costume - cache 1 208 -> ~950 slots, which
measured +6 ms TPOT for a prefill gain worth less. The loan has to be *dynamic from the start*: the cache keeps
all 1 208 slots, the stream takes the range only while a prompt is in flight, and the range is refilled
afterwards. So the phases below run P1 = loan infrastructure, P2 = spend the loan, P3 = the refill economics.

**P1 — the loan infrastructure, no behaviour change.** Allocate one bank for the borrow
region *inside* what the cache budget would have owned: `--expert-cache-gb 2.3` yields 1 208 slots today; the
partition takes the top `ceil(region_bytes / slot_bytes)` = ~340 slots (~660 MiB) out of that count and hands
them to the stream as one contiguous `[slots, N, K]` range, aliased as `[E, N, K]` for the prefill layer.
`plan` sizes the chunk from the region instead of `free`, so it can no longer decline. The cache keeps its
1 208 slots: the range is *loaned* for the duration of a prompt (`Prefill::relayout`-style) and returned after.
Gate: the startup line prints `cache N slots | borrow M MiB | ring K | drafter D`; `MISTRALRS_VRAM_DUMP`
unchanged in total; oracle byte-identical both variants; 36/36 tests; 3-round corpus within noise of M18's
baseline (4 431 / 1 153 / 45 on `doc`) with **one** buffer (behaviour must not change yet).

**P2 — spend the region on two buffers.** With the region sized by measurement rather than the formula (the
`act + act/8` terms under-count by >89 MiB, above), hold two layer buffers plus the chunk activations:
`2 x 650 + 402 + 50 + slack(measured ~280) = 2 032 MiB`, i.e. the region takes ~1 040 slots' worth (~2.0 GB) of
the budget - which the cache absorbs by having that many fewer slots *while a prompt is being prefilled*, and
gets them back afterwards. Gate: `plan` grants 2 buffers, `doc` turn-1 <= 3.7 s **and** TPOT <= 46 ms, no OOM
across a 3-round corpus (the static version OOM'd on the first turn).

**P3 — give the region back between prefills.** With no prompt in flight the region is `region_bytes /
slot_bytes` = ~340 (P1) to ~1 040 (P2) extra slots for its shape's layer, i.e. decode sees the *full* 1 208
slots again. The loan needs a flag the LRU, the hit path and `Layers::spawn_fill` all honour, plus a fence when
the stream takes the range back and a re-fill like strata's ("refilled after it", ~1.4 GiB of DMA, ~100 ms).
Gate: decode slot count restored to 1 208 with TPOT <= 45 ms, re-fill cost measured and under ~150 ms; if it
eats the gain, ship P2 with the cache permanently short and stop.

### What each phase buys, in the numbers we already have

| phase | turn-1 | TPOT | cache slots |
|---|---:|---:|---|
| M18 baseline (1 buffer, 2.3 GB) | 4 431 ms | 44.9 ms | 1 208 |
| P2 (2 buffers, cache untouched) | **~3 654 ms** | 44.9 ms | 1 208 |
| P3 (+272 loaned slots between prefills) | ~3 654 ms | <= 44.9 ms | ~1 480 |

### Risks and open questions

- **Aliasing correctness.** One `[E, N, K]` view over slots that are also addressed as `[slot, N, K]`: the
  bytes are identical, so the risk is lifetime, not layout - the stream must take the region back only after
  every reader's fence, and the cache must not write it while the stream owns it.
- **Mixed shapes.** The region is sized for the largest layer, so a smaller layer uses a prefix; the alias must
  be built for *that* layer's shape, not the region's total size.
- **Does the lookahead fill want those slots?** During decode the region is the hottest 272 slots of one layer,
  which the fill would happily overwrite. P3 needs the loan flag enforced in `Layers::spawn_fill` too, not just
  in the LRU.
- **`--pcie-frac` ring in the same arena** is a nice-to-have; it is 33 MiB and already a separate bank, so it
  can stay out of P1 if that keeps the diff small.
- **What we lose if P3 is rejected**: nothing, because P2 alone removes the M18 trade - the milestone is
  ordered so that each phase is independently worth taking.

## Why the borrow cannot pay here (measured shapes, 2026-10-21) - milestone closed

Before writing the pool, the shapes were counted from the stream's own log (seven distinct layer-buffer sizes,
because the GGUF mixes quants per layer):

| layer buffer | layers |
|---:|---:|
| 481 MiB | 15 |
| 531 MiB | 14 |
| 568 MiB | 9 |
| 368 MiB | 5 |
| 418 MiB | 3 |
| 456 MiB | 1 |
| 650 MiB | 1 |
| **total** | **48 = 23.4 GiB** |

So the cache is not one pool but seven shape groups, and both ways to make slots shareable lose more than the
borrow can win:

| layout | cost | benefit of lending the prefill run back |
|---|---|---|
| **B. uniform slot = max shape** (strata's `blob_bytes` model) | `48 x 650 = 31 200 MiB` for `23 961 MiB` of experts: **-23% capacity** at the same slot count | any layer can use the spare slots |
| **A. per-shape-group pools** (7 allocations, no padding) | 0% | a spare for the 650 MiB group serves **1 layer**; a spare for the 481 MiB group serves 15 (+16 slots/layer each) |

-23% capacity is worse than the M18 trade (231 of 1 208 slots = -19%) which already measured as a net loss
(+6 ms TPOT against 0.78 s of turn-1), and layout A's loan buys about a third of the layers a modest number of
slots. The prefill buffer is 333 slots (650 MiB), so even a *successful* loan only returns the run to the group
that happens to match it.

**Conclusion: M19 stops here.** The premise - "the same VRAM can be cache and prefill buffer" - is right in the
abstract and wrong for this model's 7-shape, 23.4 GiB expert set at 2.3 GB of cache: the bookkeeping that makes
slots movable costs more capacity than the move recovers. The second prefill buffer stays unaffordable on a
12 GB card, and M18's measured trade (cache for buffers) stays net-negative.

What the milestone leaves behind, all reusable:

- `src/moe/arena.rs` `SlotArena` with six tests - the two-watermark policy (cache from the top, prefill from the
  bottom, coldest evicted first) is correct and costs nothing to keep for a card where the cache is large enough
  for one shape group to dominate;
- the H2D instruments (`cargo test h2d_ -- --ignored --test-threads=1 -- --nocapture`) and the fact that the
  prefill copies already run at the link's 13.4 GB/s (M18);
- the measurement that the static reserve OOMs with 89 MiB of formula margin, i.e. the plan's activation terms
  under-count and must not be trusted for reservations.

Next lever for turn-1 is therefore not memory: it is the per-row compute (`attention + hc`, 0.46 ms/row) that
M17 identified, or a bigger card.

## Gates

- Oracle `frac 0 + --no-lookahead` byte-identical, with and without `--prefill-stream`; 36/36 tests.
- 3 rounds, medians; `TTFT`/`TPOT`/`reuse` comparable, `e2e` not when generated lengths differ.
- The partition must be visible in one startup line (`cache N slots | borrow M MiB | ring K slots | drafter D
  MiB`) so a run's own log explains its VRAM, the way M18's plan line does today.
