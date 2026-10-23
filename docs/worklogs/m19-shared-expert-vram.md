# M19 — one VRAM arena: the prefill stream borrows the expert cache

Status: **closed** (2026-10-22): one mechanism, four designs priced, all rejected on measurement - the last one (ring batching) was implemented and measured 3.4x slower, so it ships off by default.
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

## Is there any memory reuse left? (inventory, 2026-10-21)

After the trim removal (M18) the answer is: the *expensive* kind - moving bytes on the critical path - is
exhausted. What remains, with prices:

| item | bytes | verdict |
|---|---:|---|
| prefill buffer <-> expert cache (this milestone) | 650-1 300 MiB | refuted: 7 shape groups make movable slots cost more capacity than they return |
| prefill activations during decode | 522 MiB | no consumer: they are fp32/bf16 `[rows, hidden]` scratch, not expert blobs |
| drafter pool (MTP head, Q2_0) | 675 MiB | in use on **both** phases, no idle window to lend |
| staging ring (`--pcie-frac`) | 33 MiB | noise |
| KV cache BF16 -> int8 | 118 -> 59 MiB | real but tiny, and it changes numerics |
| dense weights BF16 -> 8-bit (`--isq` style) | 1 299 -> ~650 MiB | the only large one, but it **breaks the oracle** (the 1 299 MiB are BF16 *in the file*, so this is a requantisation, not a layout change) |
| zeroing a fresh buffer (`QCudaStorage::zeros` on every shape change) | 500 MiB x 35 per chunk | ~18 ms (~0.5% of a chunk); would need a non-zeroed allocator in candle |
| recurrent-state slots | 115 MiB each, allocated lazily | single-user already peaks at 1-2 slots, so `--max-seqs 1` buys runtime headroom, not speed |

So the remaining *large* lever is requantising the dense part, which is a numerics decision (a new oracle), and
everything else is either refuted or a rounding error. The fast wins left are not memory: TPOT's per-layer host
round trip (45 -> 39 ms) and the per-row attention/hc work (0.46 ms/row) that M17 priced.

## Reuse angles that survive the shape problem (brainstorm, 2026-10-21)

The milestone's original angle (a movable slot pool) died on seven shapes. Four other angles reuse the *same
VRAM* without needing movable slots, and two of them look worth prototyping:

### 3b. Stream the experts **through the layer's own cache slots**, in rings - no buffer at all

The layer's bank already is `[num_slots, n, k]` of that layer's exact shape, and the decode path already knows
how to DMA a *subset* of experts into slots (`DmaMiss`) and how to compute a subset with weights (`Gpu::submit`
over `hits`). So a prompt chunk could run as a sequence of rounds: DMA the next `num_slots` (25) experts of this
layer into its own slots, compute the `(row, expert)` pairs that round owns, accumulate.

| | value |
|---|---|
| rounds per layer | `256 / 25` = **10** |
| DMA total | unchanged (~499 MiB/layer, the whole expert set) |
| **VRAM saved** | **the whole 650 MiB buffer** -> **+333 cache slots (+28%)** |
| cost 1 | MoE calls per chunk: 48 -> **491** (small grouped GEMMs) |
| cost 2 | the layer's residents are clobbered; refill after the prompt = 48 x 49 MiB = **2.3 GiB ~171 ms** |

This is the user's "cache grows one way, prefill the other" idea *without* needing contiguous runs or movable
slots, because the rounds are addressed inside each layer's own bank. Cheap prototype: set `--pcie-frac 1.0`
and a **256-slot staging ring** (650 MiB, same as today's buffer) and route prompt chunks through the decode
split (raise `PREFILL_THRESHOLD` past the chunk size). That tests the *mechanism* - round overhead and the
subset compute - at the same VRAM; if it holds, the ring becomes "the layer's own slots" and the 650 MiB comes
back as cache.

### 1. Promote the hottest experts from the stream's buffer into the cache (D2D, not PCIe)

After a layer's chunk compute, the buffer holds that layer's whole expert set in VRAM. Copying the top
`num_slots` (by routing count) into the layer's cache slots is a **device-to-device** copy at ~1 TB/s:
25 x 1.95 MiB = 49 MiB -> **~0.05 ms/layer, ~2 ms/chunk**, versus the lookahead fill that fetches the same
experts over PCIe later. It fills the cache for the *next* decode with exactly the experts this conversation
uses, at copy speed, and takes load off the link that the tail and the staging ring need.

### 3c. No tail at all (KV layout)

The <=31-row tail exists only because the recurrent checkpoint must sit on a block boundary. A dense per-session
KV (strata's `conversation_kv_save(upto)`) removes the split, and with it the tail's 0.8 s; it is a KV-layout
change, not a stream change.

### Rejected on the numbers

| angle | why not |
|---|---|
| stream into a *spare* bank of the dominant shape | the spare serves only its own shape group (15 of 48 layers), and 256 slots of it = 481 MiB |
| re-layout the host stacks so all layers share one shape (strata's pack) | host 23.4 -> 31.2 GiB (+33%) **and** device slots become uniform (-23% cache capacity, M19's uniform measurement) |
| per-expert QTensor views over a byte arena | 768 tiny allocations per buffer (256 experts x 3 projections); ~5 us each = ~4 ms/layer, and it thrashes the pool |
| plain views over one arena | needs a raw-pointer `CudaSlice`, which cudarc does not expose (private fields + `Drop`) |

## Two refinements from the user, priced (2026-10-21)

### (1) Split the ring in half: compute from one half while the other DMAs

Inside one layer there is almost nothing to hide the DMA behind: per layer the chunk spends **~40 ms on the DMA**,
**~16 ms on attention + hc**, and only **~2 ms on the MoE compute** (measured, 1 725 rows). Attention *produces*
the MoE's input, so it cannot overlap this layer's expert DMA, and overlapping the DMA with 2 ms of MoE compute
saves at most 2 ms/layer (~0.1 s/chunk) - noise.

Double buffering pays only **across layers** (fill layer L+1's slots during layer L's attention), which is exactly
what the current two-buffer stream does (-16%, M18). In the ring variant that needs two layers' worth of slots -
which only the two biggest groups can nearly afford (below).

### (2) Borrow the other same-shape layers' slots: this is the idea that makes rounds disappear

A slot is only interchangeable **inside its shape group**, and the groups are unevenly sized:

| shape group | layers | slots/layer | **group slots** | group bytes | one whole layer (256 slots)? |
|---|---:|---:|---:|---:|---|
| 481 MiB | 15 | 26 | **390** | 0.72 GiB | **yes**, plus 134 spare |
| 531 MiB | 14 | 23 | **322** | 0.65 GiB | **yes**, plus 66 spare |
| 568 MiB | 9 | 22 | 198 | 0.43 GiB | no - 77% of a layer |
| 368 MiB | 5 | 34 | 170 | 0.24 GiB | no - 66% |
| 418 MiB | 3 | 30 | 90 | 0.14 GiB | no - 35% |
| 456 MiB | 1 | 27 | 27 | 0.05 GiB | no - 11% |
| 650 MiB | 1 | 19 | 19 | 0.05 GiB | no - 7% |
| | | | **1 216** | 2.3 GiB | **29 of 48 layers (60%) need no rounds at all** |

So borrowing turns the ring into a whole-layer buffer for 60% of the layers (one DMA, exactly today's copy
shape), and leaves rounds of 2-10 only for the other 40%. The 134/66 spare slots in the two big groups are enough
to *start* the cross-layer double buffering of (1) for a third of the layers.

**Price**: borrowing the bottom of a group evicts that group's residents, so a chunk that prefills every group
costs a refill of up to the whole cache - **2.3 GiB ~ 171 ms per turn** at 13.4 GB/s. Against that, the 650 MiB
buffer reservation disappears and becomes **+333 cache slots (+28%)**, which at M18's measured slope (231 slots =
6 ms TPOT) is worth about **-8 ms TPOT, i.e. ~5 s per conversation** - so the trade is positive, and it also
removes the "plan declines and prefill falls back to the CPU" failure mode entirely.

**Implementation shape**: one bank per shape group (7 allocations, `group_slots x slot_bytes`), each layer owning
a contiguous range **at the top** of its group's bank, and the prefill borrowing **the bottom** of the group it is
currently prefilling. No views are needed - everything stays slot-addressed inside the group's own `QTensor`
(`Bank { first }` plus the kernels' `local = slot - first`), which is why this version works where the movable-slot
arena could not. What is still missing is the refill policy (the evicted slots come back as empty) and the
per-group eviction bookkeeping.

## Borrow policy: the current layer is the primary (decided 2026-10-21)

Not all borrowed slots are equal. Right after a chunk, the decode steps consume **the same layer's** experts for
the new tokens, so layer L's own residents are the hottest thing in the cache - evicting them to make room for L's
own prefill would defeat the point. The policy therefore is:

1. **empty slots first** (nothing resident, free to use);
2. **other layers of the same shape group**, coldest first;
3. **only then** L's own slots, and those are the ones marked `borrowed` for the earliest return.

"Borrowed" is a per-slot mark: the slot is ours from the claim until `release`, it is skipped by the hit path, the
LRU and `spawn_fill`, and on release it goes back as **empty** (the cache refills, it does not restore).

At a 4 GiB pool this is enough to never touch the current layer for almost every layer:

| shape group | group slots | L's own | borrowable from others | 256 without touching L? |
|---|---:|---:|---:|---|
| 481 MiB | 675 | 45 | 630 | **yes** |
| 531 MiB | 574 | 41 | 533 | **yes** |
| 568 MiB | 342 | 38 | 304 | **yes** |
| 368 MiB | 295 | 59 | 236 | short 20 |
| 418 MiB | 156 | 52 | 104 | short 152 - rounds |
| 456 MiB | 47 | 47 | 0 | rounds |
| 650 MiB | 33 | 33 | 0 | rounds |

So **42 of 48 layers (87.5%) prefill without evicting a single one of their own residents**, 5 layers (368 group)
sacrifice ~20 of their own slots, and 5 layers of the two single-layer groups run in rounds. What decode gets back
is exactly the locality that matters: the experts the prompt just routed stay resident for the first decode steps,
which is where the turn's TTFT tail is paid.

Bookkeeping it needs, on top of `SlotArena`: a `layer` tag per slot (to know what "other layers" means) and a
`borrowed` set per claim (to skip in hit/LRU/fill and to release). Both live where `Slot` already lives, next to
`expert`/`buf`/`last_use`.

## When the borrow falls short: the ring takes over (decided 2026-10-21)

The ring is not a separate mode, it is **what is left over**: a prefill round addresses *any set of slot indices*
inside the group's bank (the kernels take `local = slot - first`, so contiguity is not required), which means the
ring size is simply `empty + borrowed-from-other-layers + the layer's own slots`, and the rounds are
`ceil(256 / ring size)`.

At a 4 GiB pool:

| shape group | layers | own | borrowable | ring | rounds | extra MoE calls |
|---|---:|---:|---:|---:|---:|---:|
| 481 MiB | 15 | 45 | 630 | 675 | **1** | 0 |
| 531 MiB | 14 | 41 | 533 | 574 | **1** | 0 |
| 568 MiB | 9 | 38 | 304 | 342 | **1** | 0 |
| 368 MiB | 5 | 59 | 236 | 295 | **1** | 0 (20 of its own slots are used, i.e. evicted) |
| 418 MiB | 3 | 52 | 104 | 156 | **2** | +3 |
| 456 MiB | 1 | 47 | 0 | 47 | **6** | +5 |
| 650 MiB | 1 | 33 | 0 | 33 | **8** | +7 |

So **38 of 48 layers prefill in one pass**, 3 layers in two, and only the two single-layer groups (the 456 and
650 MiB ones) need 6-8 rounds. The total added cost is **15 extra small MoE calls per chunk** (+31% calls, but the
MoE compute is ~2 ms/layer, i.e. ~1% of a chunk) plus ~2 500 small per-expert copies instead of whole-blob copies
for those 10 layers (~1%).

The three-tier policy and the ring therefore compose without a switch: borrow what is free, then what other layers
hold, use the layer's own slots only as ring capacity, and run as many rounds as the ring needs. Nothing evicts a
layer's own residents unless the group cannot cover one layer without them, and every borrowed slot is marked for
return and comes back empty.

## Sizing: 2 GB, 3 GB, 4 GB (2026-10-21)

The same analysis at each cache size, with the borrow in place (the 650 MiB buffer reservation is gone, so the only
thing that still needs *genuinely* free VRAM is the chunk's activations plus the measured slack, ~868 MiB at chunk
8 192):

| cache | slots | 481-group slots / borrowable | layers needing >1 round | extra MoE calls | free VRAM left | feasible |
|---|---:|---|---:|---:|---:|---|
| 2.0 GB | 1 043 | 330 / 308 | 19/48 | 49 | ~2 290 MiB | yes |
| 2.3 GB (today) | 1 216 | 390 / 364 | 19/48 | 42 | ~1 983 MiB | yes |
| **3.0 GB** | **1 579** | 510 / 476 | 19/48 | 37 | ~1 266 MiB | **yes - the ceiling** |
| 3.5 GB | ~1 840 | 595 / 555 | 19/48 | 31 | ~754 MiB | no (needs ~868) |
| 4.0 GB | 2 122 | 675 / 630 | 5/48 | 15 | ~242 MiB | **no** |

Two things fall out of it:

1. **The number of multi-round layers does not move between 2.0 and 3.0 GB** (19 of 48). It is set by the *small*
   shape groups, whose total scales with their own layer count: the 456 and 650 MiB groups are single layers (23
   and 16 slots at 2 GB, 35 and 25 at 3 GB), and the 418 MiB group is three. What improves with size is the *round
   count of those few layers* (16 -> 11, 12 -> 8, 4 -> 3), not which layers.
2. **The 29 layers of the three big groups get single-round prefills from 2.0 GB upward**, because 330 and 280 group
   slots already exceed the 256 a layer needs. So the ring machinery is needed only for the 19 small-group layers
   and is cheap there (49 extra small MoE calls per chunk at 2 GB, ~30 ms, ~1%).

**4 GB is not reachable on this card** with the borrow design alone: the activations are `[rows, hidden]` float
tensors, not expert-shaped slots, so they cannot be carved from the slot pool; ~868 MiB of real free VRAM has to
stay, and 4 GB leaves 242 MiB. **3.0 GB is the practical ceiling**, and it buys 1 579 slots - **+30% over today's
1 208** - with the same round profile as 2.3 GB.

## Overlap design for ring batching, and the measurement it hinges on (2026-10-21)

### The model

Per layer of a 1 725-row chunk: **DMA 499 MiB = 40 ms** at the link's 13.4 GB/s, **attention + hc = 16 ms**, **MoE
compute = 2 ms**. The floor is therefore `max(40, 18) = 40 ms/layer = 1.92 s/chunk`, and the measured per-layer
times imply how much of it we are losing:

| measured | ms/layer | of which "DMA segment" | implied copy rate |
|---|---:|---:|---:|
| 1 buffer, with the trim | 94 ms | 76 ms | 6.4 GB/s |
| 1 buffer, no trim | 81 ms | 63 ms | 7.7 GB/s |
| 2 buffers, no trim | 78 ms | 60 ms | 8.1 GB/s |

So even with a whole layer of lead time (two buffers) the copy runs at **8 GB/s, not 13.4**. Overlap is not the
limiting factor yet - the copy *rate* in the pipeline is, and that is what a smarter ring pipeline cannot fix by
itself.

Two candidate causes, and they lead to different designs:

| hypothesis | test | if true |
|---|---|---|
| **the cache fill shares the link** during streamed prefill (`spawn_fill` issues PCIe copies on the same copy stream) | `NAIWA_NO_PREFILL_FILL=1` skips the chunk path's fills (probe written) | remove or defer the fill, keep the link for the buffer: ~40-45 ms/layer, ~1.9-2.2 s/chunk |
| **the copy contends with attention for HBM bandwidth** (the DMA writes 499 MiB to HBM while attention streams KV) | `nvidia-smi utilization.memory` during a chunk, and the same chunk with attention artificially cheap (short prompt, large chunk) | no overlap trick helps; the floor is 50-60 ms/layer and the ring design should optimise *bytes*, not overlap |

### The minimal design: rename `layer` to `round` and keep everything else

The three-stage sketch below is more machinery than the problem needs. `PrefillStream` already *is* a
two-slice ping-pong: `bufs[0..stride]`, each with its own `released`/`ready`, and `forward(layer)` prefetches
`layer + 1` before computing `layer`. The only thing that makes it layer-shaped is the index it uses.

**Minimal version: index the ping-pong by a *global round counter* instead of by layer.**

```
rounds: (layer, expert subset) in layer order, across the whole chunk
slice  = round % 2                      # the two slices are the group's slots, halved
load(round + 1) into slice ^ 1          # already what `stride == 2` does, one level down
compute(round) from slice               # already what `forward` does, on `hits` of that subset
```

That is it. No new fence types, no per-ring bookkeeping, no explicit cross-layer stage: because the round counter
runs across layers, **the last round of layer L prefetches the first round of layer L+1 by construction**, which
is exactly the cross-layer lead the three stages below were building. The two slices alternate globally, so the
copy engine never waits for a layer boundary.

Sizing: `ring = group_slots / 2`, `rounds per layer = ceil(256 / ring)`.

| shape group | group slots | ring (per slice) | rounds/layer | DMA per slice | compute per slice |
|---|---:|---:|---:|---:|---:|
| 481 MiB | 510 | 255 | 2 | 36 ms | 9 ms |
| 531 MiB | 420 | 210 | 2 | 33 ms | 9 ms |
| 568 MiB | 252 | 126 | 3 | 21 ms | 6 ms |
| 368 MiB | 220 | 110 | 3 | 12 ms | 6 ms |
| 418 MiB | 117 | 58 | 5 | 7 ms | 4 ms |
| 456 MiB | 35 | 17 | 16 | 2 ms | 1 ms |
| 650 MiB | 25 | 12 | 22 | 2 ms | 1 ms |

Because the slices alternate, in steady state a layer costs `max(DMA 40, compute 18) = 40 ms` (the small groups
just have more, smaller slices of the same totals): **1.92 s/chunk at 13.4 GB/s, and ~2.9 s if the copy stays at
the measured 8 GB/s** - which is the same fork in the road as before, and the reason the cheap NAIWA_NO_PREFILL_FILL
probe comes first.

Two implementation notes, both small:

- the per-round expert subset is exactly what the cache already computes for a decode step (`hits` with their
  weights), so a round's compute is a `Gpu::submit`-shaped call, not a new kernel path;
- the slices are borrowed group slots, so "who owns them" is the `SlotArena` watermarks plus the `borrowed` mark -
  no view over the bank is needed, since a slice is just a set of slot indices.

### The fuller design (kept for the record, superseded by the above)

Three stages, each adding one lead:

1. **Per-round ping-pong inside the layer (two rings per group, ring = 128 slots).** Round *i* computes from ring
   A while the DMA fills ring B with round *i+1*; the next DMA is enqueued at the **end of round *i*, not at the
   end of the layer** (that is the only host-side discipline the current code lacks - it enqueues one layer ahead).
   Two rings of 128 slots need 256 group slots, which every big group has at 2.3 GB or more.
2. **Cross-layer lead for the first round.** Layer L's last round runs while layer L+1's first round is being
   filled, so the copy engine never idles at a layer boundary. This is the ring's version of the second buffer
   the stream has today. Same-group layers share the group's slots, which is why *two* rings (not one) are needed.
3. **Round granularity fences.** Each ring's completion and release must be per-ring, not per-layer, or the
   ping-pong deadlocks: the copy of B waits on A's readers, and A's next fill waits on B's readers. `CopyDone` /
   `ReadFence` already exist per buffer today; this is about carrying them per ring.

### What it would be worth

If the copy reaches 13.4 GB/s, ring mode with the three stages lands at ~40 ms/layer (**1.9 s/chunk, -50%** from
today's 3.88 s), using **borrowed** slots (no 650 MiB reservation) and with 38 of 48 layers in a single round and
the other 10 in 2-8 rounds. If the copy stays at 8 GB/s, the same design lands at ~60 ms/layer (2.9 s) and the
remaining loss is HBM, not scheduling - in which case the right move is fewer bytes (which is what `--expert-cache-gb
3` plus the borrow already gives, by raising the hit rate).

## What the chunk is actually bound by (measured, 2026-10-21) - the overlap question closes

Four measurements, run in this order, killed the overlap design as a *speed* lever:

| probe | result | reads as |
|---|---|---|
| `NAIWA_NO_PREFILL_FILL=1` (skip the cache fill during a streamed chunk) | turn-1 3.94 s -> **4.03 s** | the fill is *not* competing for the link |
| `nvidia-smi` during a chunk | `utilization.gpu` **100%**, `utilization.memory` **16-24%**, PCIe Gen3 | the chunk is **SM-bound**, not HBM- or PCIe-bound |
| `MISTRALRS_LAYER_DBG=1` (new: per-layer time inside the model forward) | 1.0-2.6 ms per layer, two 84/167 ms outliers on layers 0 and 3 | the host only measures *enqueue* time (CUDA is async); the outliers are first-touch module loads. Not usable to decompose GPU time |
| same 1 725-token prompt, one chunk vs two | **3.22 s vs 7.07 s** | the cost is **per chunk, not per row**: two chunks of the same 1 725 rows cost 2.2x, so the pass over each layer's expert set dominates and the row count barely matters |

Consequences:

- The minimal overlap design is **already implemented** at one round per layer: today's `stride = 2` stream prefetches a whole layer ahead, `dma_wait` is ~700 ns, and the copy is hidden. Making rounds smaller cannot help, because there is no exposed DMA to hide.
- The chunk's 3.2 s is the **expert-set pass**: 48 layers x (499 MiB of experts + their quantized compute), and it is SM-bound. Overlap, buffer counts, ring sizes and borrow policies cannot reduce it; only *fewer experts per chunk* can, which is bounded by the cache's residency (9.9% today, ~17% at 3 GB) and by the fact that a chunk saturates the 256 experts above ~26 rows.
- So the ring-batching design stays a **memory** instrument (it removes the 650 MiB reservation and lets the cache grow), not a speed one, and its cost/benefit is the one priced earlier: roughly neutral on turn-1, positive on capacity.

The overlap question this document spent three sections on is therefore answered: **there is nothing to overlap.** The remaining prefill levers are the MoE kernel's efficiency on a saturated expert set (a quantized grouped-GEMM question, the one strata answers with MMQ/fused kernels) and the per-row attention/hc tax.

## Ring batching implemented and measured: it is 3.4x slower (2026-10-22)

Built it, ran it, and the plan's prediction was too optimistic by an order of magnitude.

**What exists now** (`NAIWA_PREFILL_RINGS=1`, off by default):

- `src/moe/rings.rs`: the round planner, six unit tests - rounds partition the routing exactly (no pair
  computed twice, no pair lost, weights preserved), respect `own + lent` capacity, assign the first rounds to the
  layer's own slots and the rest to a lender, and keep the largest round within 2x of the average by greedy
  longest-processing-time over expert popularity;
- `src/moe/rings_exec.rs`: the executor - rents the layer's slots, DMAs each round's experts from the pinned host
  stacks into them, computes the round's `(row, expert)` pairs with the same fused kernel the decode hits use, and
  sums the rounds.

**Two ownership bugs on the way, both the same shape**: `wait_for_readers` spins until `Arc::strong_count == 1`,
so any clone kept alive while waiting deadlocks. The first version cloned the buffer out of `Slot::storage` and
spun forever; the second cleared `storage` but left `Slot::buf` (the published contents) holding a reference and
spun again. The fix is to drop *everything* that can hold the buffer - `storage` (taken), `buf`, `expert` -
before waiting. The W105 comment in `dma_miss.rs` warns about exactly this.

**The measurement** (3-turn `doc`, cache 2.3 GB, own slots only = 10 rounds per layer):

| | turn-1 | turn-2 | turn-3 |
|---|---:|---:|---:|
| prefill stream (baseline, same binary) | **4.02 s** | 1.30 s | 2.29 s |
| ring batching, 10 rounds/layer | **13.48 s** | 1.30 s | 1.01 s |

So **an extra round costs ~1.05 s per chunk** across the 48 layers (9 extra rounds = +9.46 s). The reason is that
each round is a full stop: a `CopyDone` wait before the round's compute, then a small GEMM, 10 times per layer,
480 times per chunk, with no overlap between the DMA of round i+1 and the compute of round i.

**Consequence, and why borrowing cannot save it**: even if a lender made every layer 1 round (the best case), the
cost would be the baseline; at the 2 rounds a 2.3 GB pool affords, the extrapolated cost is ~5 s - still worse
than the 4.02 s stream. Ring batching therefore **cannot pay for speed on this card**; it only pays in memory
(-650 MiB of buffer), and M18 already measured what that memory is worth (a net loss at 231 slots).

**Kept in the tree, off by default**, because the planner is correct, tested, and the executor is the shortest
demonstration of the mechanism - it is the thing to reach for if a future card makes slots plentiful enough for
1-round prefills *without* a dedicated buffer.

## Gates

- Oracle `frac 0 + --no-lookahead` byte-identical, with and without `--prefill-stream`; 36/36 tests.
- 3 rounds, medians; `TTFT`/`TPOT`/`reuse` comparable, `e2e` not when generated lengths differ.
- The partition must be visible in one startup line (`cache N slots | borrow M MiB | ring K slots | drafter D
  MiB`) so a run's own log explains its VRAM, the way M18's plan line does today.

## Reopening the design: what a round really costs, and the instrument that cannot say (2026-10-22)

The ring measurement left one number unexplained: 9 extra rounds cost 9.46 s, i.e. **21.9 ms per layer-round**,
while a round only moves ~25 experts (47 MiB, 3.5 ms at 13.4 GB/s). If that 18 ms were removable overhead, the
design would change completely - so before discarding rings, price the gap. It turned out the gap is *compute*,
and the instrumentation hid it.

**The instrument is broken for this question.** `PrefillStream::forward` reports

```
compute = experts.forward(...).elapsed()   // returns a Tensor: this is the *enqueue*, not the execution
copy    = buf.queued.elapsed()             // `queued` is a timestamp from the enqueue of the copies
```

Both are stopwatch readings around *asynchronous* work. One 1725-row turn-1 chunk, all 48 layers:

| | reported |
|---|---:|
| `compute` sum over 48 layers | **0.161 s** |
| `copy` sum over 48 layers | 0.162 s |
| bytes | 23 961 MiB (23.4 GiB) |
| turn-1 wall | **4.04 s** |

48 layers of 256-expert MoE cannot finish in 161 ms; the number is the launch time. Likewise the derived rates are
meaningless: 45 of the 48 layers report 2000-6900 GB/s, i.e. *481 MiB copied in 100 µs*. Whatever those lanes did,
it was not PCIe.

**But three layers are honest**, and they are the ones that had to *wait* for their copy - so the wait, not the
enqueue, dominates the reading:

| layer | bytes | `compute` = `copy` | rate |
|---|---:|---:|---:|
| 0 | 531 MiB | 57.4 ms | 9.7 GB/s |
| 1 | 368 MiB | 37.3 ms | 10.4 GB/s |
| 17 | 568 MiB | 50.8 ms | 11.7 GB/s |

That is the real cost of streaming one layer: **50-57 ms for 481-650 MiB at 10-12 GB/s**, against the 13.4 GB/s
the H2D calibration measured. And 2.93 s / 48 layers = **61 ms per layer** of fixed chunk cost - so per layer,
PCIe (≈55 ms) and compute (≈61 ms) are *the same size*. The chunk is not "SM-bound with PCIe idle"; the two are
balanced, and a layer's wall time is whichever finishes last. 23.4 GiB / 4.04 s = 6.2 GB/s average, i.e. the bus
is not saturated but no layer is idle either.

**What that means for the ring design, item by item:**

- **Rounds multiply compute and not bytes.** A round moves the same total experts (each exactly once) but issues
  10x the MoE calls. In a chunk where compute and PCIe are balanced at ~57 ms/layer, doubling the compute side
  doubles the layer. Measured: 3.4x. The design cannot be rescued by a better round *plan* - the plan is already
  pair-optimal (six tests); the loss is in the *execution* granularity.
- **The cost per round is the decode kernel.** Rings execute with `fused_decode` - the kernel tuned for
  `rows` of 1-4 - at prefill width, because the cache's slots are scattered `Arc<Bank>`s and the grouped prefill
  GEMM needs one contiguous stack. So the mechanism's compute penalty is structural: it forces the *decode* kernel
  for *prefill* work.
- **Borrowing slots pays in bytes, not compute - and the bytes it saves are small.** A resident expert skips its
  copy: 25 slots / 256 experts ≈ 10% of a layer's PCIe traffic ≈ 5.7 ms/layer ≈ **-0.27 s/chunk**. But serving the
  resident experts from the cache bank while the rest come from the stream buffer means a *second* MoE call per
  layer - the same launch penalty that just lost 9.46 s. Net: zero at best.
- **Therefore: no ring variant can win on this architecture.** Not because the round count is too high (a lender
  would fix that) but because (a) the chunk is compute/PCIe balanced, so extra compute is pure loss, and (b) the
  only kernel that can consume scattered slots is the wrong shape for prefill.

**The one design this leaves open**, recorded so it is not re-derived: rings become viable only if a *single*
grouped kernel can consume a scattered expert set - i.e. a grouped GEMM with an expert-index indirection into the
slots. That is a fork-side kernel (the same place the remaining turn-1 gap points, M20's grouped GEMM/MMQ work),
and it would make all three memory designs (rings, borrowing, unified slots) suddenly affordable. Until then, the
650 MiB buffer is the price of one contiguous pass over the expert set, and it is not negotiable.

**Instrument todo, before any further transport work**: `compute` must be measured with a sync (or a recorded
event) and `copy` from the layer's own completion event, not from a stale `queued` timestamp. The current pair of
numbers cannot separate PCIe from compute, and both of this milestone's remaining questions are exactly that.

## The contiguous-round design: right idea, and the kernel refuses it (2026-10-22)

The v1 diagnosis ("rings force the decode kernel") suggested a better design, and it is worth recording because
it is the *only* version of this milestone's mechanism that has the right cost shape:

**Borrow one layer's whole bank, not a few slots.** A layer's `Bank` is one contiguous `[slots, n, k]` QTensor
per projection (`copy_stream::alloc_bank`) - exactly the layout of the model's expert stack. So a round over a
whole bank is literally a `MoEExperts::from_qtensors(bank.gate, bank.up, bank.down, slots, top_k, ...)`, and
`forward(xs, weights, ids)` runs the **prefill grouped kernel** over it. The routing table for a round is just
the slot index - `ids`, not expert ids. That removes v1's structural penalty: the grouped GEMM costs the true
pairs, so 11 rounds of 25 experts should cost one pass plus 11 launches, instead of one pass times the padding
factor.

**It is also what makes borrowing worth doing at all**: with 25 slots a layer needs ~10 rounds, but each round is
then a *contiguous* 25-expert stack, and rounds can run on slots borrowed from any same-shape layer (`RoundSlots`
in the planner already models it) with double buffering to overlap the next round's DMA.

**Implemented, and it does not run.** Two attempts:

1. build a `MoEExperts` per round with `top_k` = that round's routing width - the chunk *completes* (48 layers x
   8-9 rounds logged) but the turn ends in a generic `Internal server error`;
2. build one per layer (the bank does not change between rounds; only the routing does) and pass the model's
   `top_k` - fails immediately with what attempt 1 was swallowing:

```
RINGS_ERR layer 0: moe experts forward
moe experts fast grouped
DriverError(CUDA_ERROR_ILLEGAL_ADDRESS, "an illegal memory access was encountered")
```

An illegal address poisons the CUDA context, which is why attempt 1 only showed up as a late, message-less
engine error. So the grouped fast path will not take a bank-backed stack and per-round routing as-is. The
suspects are all inside the fork's kernel rather than the mechanism: the bank's per-expert stride/alignment
versus what the grouped kernel derives from the stack's own dims (the *decode* path indexes these banks fine,
one expert at a time), and whether `topk_ids` width must equal the model's `top_k` rather than the round's.

**Status: reverted.** The v1 executor (`NAIWA_PREFILL_RINGS=1`) is what stays in the tree; this design is
recorded with its blocker because it is the one to pick up when a fork-side grouped kernel accepts an expert
indirection - the same kernel work the remaining turn-1 gap points at. The cost shape it promises (one pass plus
launches) is the only shape that can beat the 650 MiB buffer.

**What it would take to finish**: a standalone CUDA repro that builds a 27-slot bank from a real IQ stack and
runs `MoEExperts::forward` on 1725 rows, so the illegal address can be bisected without the server, the prefix
cache and the drafter in the same process. That repro is also the microbenchmark that decides the design: 25
experts vs 256 on the same rows gives the per-launch cost that all of this hinges on.

## Compared with Strata: what it borrows, and what is idle in ours (2026-10-22)

Strata solves the same problem with three mechanisms, and its docs name them: a **streamed ring** of expert slots,
a **loan** of expert-cache room for the prompt path's *buffers*, and a **chunk chosen to fit both**.

- *Ring*: "Since engine 0.1.39b (#583) the streamed ring is sized in bytes for the pack (a native pack with bigger
  blobs gets fewer ring slots and keeps more cache slots) and `--prefill auto` picks the largest chunk that keeps
  that ring full." Its ring is **tens of MiB**, because the prompt's experts are multiplied by the same quantized
  MMQ kernel the decode uses - the ring is a *slot* of experts, not a layer.
- *Loan*: "The prompt path **borrows room in the GPU's expert cache for its buffers** and puts those experts back
  after the prompt."
- *Chunk*: `--prefill auto` trades ring slots for chunk size only when the old rule capped the chunk at 4,096
  tokens; "a smaller ring only slowed it" - shrinking the ring to keep cache slots measured **-12%** on the
  Coder. That is our M18 result, measured independently.

**Where we differ, and why our ring must be 650 MiB.** Our prompt path runs a *grouped* MoE kernel over one
contiguous 256-expert stack, so the streamed unit is a whole layer. Strata's kernel takes an expert index list, so
its unit is one slot and its ring can be small. That single difference explains the whole memory gap: 650 MiB vs
tens of MiB, and it is the same kernel gap M19's last design hit (`CUDA_ERROR_ILLEGAL_ADDRESS`).

**The thing we have been ignoring: during a chunk our expert cache is completely idle.** The chunk runs from the
layer buffer; the cache's 2.3 GB is untouched (M17 measured that filling it during a chunk changes nothing:
`NAIWA_NO_PREFILL_FILL=1` 3.94 vs 4.03 s). Meanwhile 522 MiB of activation scratch is squeezed into the leftover
free VRAM, and `reserve` sizes it for the **cap** (8192 rows) although the chunks actually run are 1725-1900.

Two ways to reuse that, in increasing order of work:

1. **Size the activation reservation for the chunk that runs.** `activation_bytes(chunk) = 160 MiB + 170 MiB +
   23.4 KiB x rows`; at 8192 it is 522 MiB, at our 1725 rows it is 369 MiB. Letting the reservation follow the
   plan instead of the cap frees **~150 MiB** for nothing.
2. **Move the activation scratch into the cache's VRAM during the chunk, and put the displaced experts back.**
   The cache is idle for exactly that window, so borrowing it costs nothing while the chunk runs; afterwards the
   ~278 slots the scratch covered (522 MiB / 1.88 MB) are refilled (45 ms of H2D, or simply re-read by the next
   decode). The arithmetic: prefill's own footprint drops from 650 + 522 + 192 = 1364 MiB to 650 + 192 = 842,
   which at 2337 MiB free moves the cache ceiling from **2.3 GB to ~3.7 GB (+60% slots, 1208 -> ~1930)**.
   This is *not* M18's rejected trade: M18 spent cache permanently on a second weight buffer, which does nothing
   for decode. Here the cache is only unavailable during the prompt, and decode's steady state is untouched.

Both are independent of the kernel work, and (2) is the largest memory win this card still has. It needs an f32
arena over the cache's banks - a `Tensor` built on an existing device pointer - which is a much smaller job than
the grouped-kernel indirection that blocks everything else.

## The measurement the design was missing: slots do buy TPOT (2026-10-22)

Every memory design in this milestone was rejected on *turn-1* cost, and none of them had the number that
decides the direction: how much TPOT a slot is worth. Two runs of the `short` conversation (first prompt 1 034
tokens, so one chunk in both configs - the chunk does not change), 64-token answers:

| `--expert-cache-gb` | slots | turn-1 TPOT | turn-2 TPOT | turn-3 TPOT | turn-1 TTFT |
|---|---:|---:|---:|---:|---:|
| 2.3 | 1 208 | 62.9 ms | 41.5 ms | 33.9 ms | 1.30 s |
| 3.0 | 1 579 | **56.5 ms** | **37.3 ms** | 34.7 ms | 1.32 s |

**+30% slots is worth about -10% TPOT** on the turns where the cache is not yet warm, and nothing on TTFT. The
gain fades by turn 3 (33.9 -> 34.7), where reuse is 0.79-0.82 and the working set already fits - which is exactly
where our gap to Strata is smallest.

**And the same run shows why the cache cannot simply be raised.** At 3.0 GB the planner reports
`prefill: the scheduler's prompt step is capped at 1280 tokens`: the chunk and the cache are paid from one budget,
so every MiB given to the cache is taken from the chunk. A 1 725-row prompt then needs **two chunks**, and M17
measured two chunks at 2.2x the cost of one. So a static raise buys -10% TPOT and pays ~2x turn-1 for `doc` and
`long`. That is the trade M18 rejected, now with a number attached to both sides.

**What that means for the design of record** (the two-sided allocator above): it is the only shape that gets both,
and its closing argument was wrong in one specific way. M19 closed because "seven shape groups make movable slots
cost more capacity than they return" - but that refutes *unifying slot sizes*, which the design does not need. The
borrow region is a **byte range**: the stream writes every byte it borrows, so what it overlays does not care what
shapes those banks were, and the eviction cost is counted in slots (1.4-2.5 MB each), not in lost capacity.

So the design to build:

1. **One slab, per-layer banks carved from it** (unchanged shapes, unchanged capacities), plus a borrow range at
   the slab's low end.
2. **The cache is sized as `free_at_load + chunk footprint`** - at today's numbers `650 + 369 + 192 = 1 211 MiB`
   for a 1 725-row chunk (the activation term must follow the *plan*, not the 8 192 cap that overstates it by
   153 MiB), i.e. **`--expert-cache-gb` ~3.5 instead of 2.3, about +52% slots**.
3. **At prompt time the stream takes the range**, evicting whatever slots live there (~640 at 1.88 MB each, marked
   empty), runs the chunk, and returns it; decode refills them as it would after any eviction.
4. Numerics unchanged, so the oracle stays byte-identical: the chunk's bytes are the same bytes at a different
   address.

**Why it should pay**: the TPOT table above prices +52% slots at roughly -15% TPOT (42 -> ~36 ms on `doc`, at or
past Strata's 39), i.e. ~0.4 s over a 64-token turn, against M19's P3 estimate of ~100 ms of refill DMA per prompt.

**What has to be measured before it is worth building**: the refill's real cost per turn (M19's estimate was
1.4 GiB / ~100 ms, never measured), the chunk's true activation peak (the formula under-counted by >89 MiB at
2 816 rows, and `HEADROOM` is 192 MiB), and whether the -10% holds for `doc`/`long` once they are back to one
chunk under the loan. The slab itself is mechanical: `alloc_bank` already takes a length and a device, so it needs
a base offset and a bump/free list, not a new allocator.

## Feasibility check: the cache cannot simply be raised, and why (2026-10-22)

Before judging the slab design, the first question is how much cache the *current* design can afford, since
`--expert-cache-gb 2.3` was chosen in M10 and has not been re-tuned after the drafter (675 MiB), the checkpoint
capture and the trim removal moved the memory map. Bisected by starting the engine and reading the plan:

| `--expert-cache-gb` | slots/layer | chunk cap | prefill stream | `doc` turn-1 |
|---|---:|---:|---|---:|
| 2.3 (today) | 27 | 8 192 | on | **4.0 s** |
| 2.8 | 33 | 8 192 | **off** | **132 s** |
| 2.85 | 33 | 7 168 | off | - |
| 2.95 | 35 | 2 304 | off | - |
| 3.0 | 35 | 1 280 | off | - |

Two things fall out, and the second one is the important one:

1. **There is no free headroom.** +0.5 GB (27 -> 33 slots, +22%) already breaks the stream: at 2.8 GB the plan
   says `off (650 MiB layer buffer + 522 MiB activations + 192 MiB headroom does not fit in 1408 MiB free)`, and
   the check also adds `act/8`, so the miss is ~20 MiB. 2.3 GB is not a conservative choice, it is the working
   point.
2. **The failure is silent and catastrophic.** Nothing crashes: the stream turns itself off, every layer's MoE
   falls back to the CPU, and turn-1 goes 4.0 s -> **132 s** (33x) with only a WARN in the log. Any design that
   grows the cache has to treat "the stream must keep its buffers" as an invariant, enforced loudly.

That also explains the earlier `short` result: at 3.0 GB that conversation still ran one chunk (1 034 rows is
under the new 1 280 cap), which is why its TPOT improved cleanly and its TTFT did not move - it was the only
config where "more cache" did not also mean "smaller chunk".

So the judgement on the slab design stands, with the ordering fixed by this run:

- the loan is the **only** way to grow the cache, because the 650 MiB buffer is already inside the same budget;
- **Phase A (overlay the 650 MiB layer buffer only)** is the version to build: it returns the same +22-30% slots
  the 2.8-3.0 points show, while keeping the chunk at 8 192. Measured value on the slot->TPOT curve: +30% slots
  is -10% TPOT (`short`: 62.9 -> 56.5 ms, 41.5 -> 37.3), i.e. `doc` 42 -> ~38 ms;
- **Phase B (the 369-522 MiB of activations)** needs candle's allocator to hand out arena memory, so it is a
  much larger job for maybe another 5%;
- and **nothing in this direction touches the turn-1 gap** (+60% against Strata): that is the kernel side.

**Kill criteria, written before building**: turn-1 within 200 ms of today, TPOT <= 38 ms, refill under 150 ms per
prompt, and the startup check must refuse to start rather than silently drop the stream.

## Step 0 done: the peak instrument and a loud failure (2026-10-22)

Two small things landed before any allocator work, because both later decisions rest on them.

**`NAIWA_PREFILL_PEAK=1`** samples free VRAM at every layer of a chunk and reports the footprint against the
planner's estimate. First real numbers (`doc` turn-1, one chunk, cache 2.3 GB):

```
prefill peak: chunk 1725 rows, 48 layers, 1378 MiB used (1983 MiB free at start, 605 MiB low water);
              layer buffer 650 MiB, planner estimate 1364 MiB (act 522)
```

So at this chunk size the formula is **almost exactly right** (1364 vs 1378, 14 MiB under): M19's "under-counts
by >89 MiB" was a 2 816-row chunk, not this one. The interesting number is the other one - **605 MiB of true low
water**, i.e. what the plan leaves unused at the deepest point of a chunk. That is the space every "grow the
cache" idea wants, and it is ~600 MiB, not 0.

**`NAIWA_PREFILL_STRICT=1`** turns "the stream cannot get its buffer" into an error instead of a WARN followed by
the CPU taking the chunk. It earns its keep immediately: with it set, the next experiment fails loudly where the
previous one silently took 132 s.

**Sweeping the cache with both flags on** (the point of the exercise):

| `--expert-cache-gb` | plan | first chunk | outcome |
|---|---:|---|---|
| 2.3 | 8192 | 1725 rows, used 1378 MiB, low water 605 | **works**, turn-1 4.0 s |
| 2.6 | 4864 | 1696 rows, used 1442 MiB, low water **222 MiB** | chunk completes, then `CUDA_ERROR_OUT_OF_MEMORY` in `moe experts forward` |
| 2.8 | off | - | stream refuses the plan (misses by ~20 MiB) |

The 2.6 row is the finding: **the chunk is not what breaks.** Its own peak finished with 222 MiB to spare, and
the OOM came afterwards - so the unmodelled consumer is the post-chunk work (the MTP verify step, the recurrent
pool, the prefix-cache checkpoint), not the prefill. The code's 512 MiB `head` floor exists for exactly this
("192-227 MiB of slack OOM'd a 1723-token chunk twice"), and it is why `--expert-cache-gb 2.3` is the working
point rather than a conservative choice.

**Consequence for the loan design**: its sizing cannot be "chunk footprint + margin". It has to be "what the whole
request needs at its deepest point", which includes the decode/verify side. The peak instrument therefore needs one
more sample point - the decode step - before Phase A can size anything. That is the next small piece of work, and
it is cheap: the same free-VRAM sample at the top of a decode step, reported against the same baseline.

## The other half of the peak: decode bottoms out at the same place (2026-10-22)

Added the same sample to the decode path (`NAIWA_PREFILL_PEAK=1`, once per step on layer 0, `src/vram_peak.rs`).
One `doc` turn at cache 2.3 GB:

```
prefill peak: chunk 1696 rows, 48 layers, 1378 MiB used (1985 MiB free at start, 607 MiB low water)
decode peak:  64 steps, 601 MiB lowest free VRAM (601 MiB now)
```

**The chunk and the decode bottom out within 6 MiB of each other.** That is the answer to "how much is the cache
allowed to grow": none of that ~600 MiB is spare. It is what the decode side needs once the chunk is done - the
MTP verify step, the recurrent pool, the checkpoint capture and the pool's own slack - and it explains the 2.6 GB
OOM exactly: that run finished its chunk with 222 MiB left, i.e. ~380 MiB short of what decode needs, and the OOM
landed in the post-chunk work rather than in the chunk.

So:

- **growing the cache out of today's free VRAM buys at most ~200 MiB (+7% slots)**, which is inside the noise of
  the -10%-per-30% curve. That avenue is effectively closed, and it is closed by a *measured* requirement rather
  than by the formula's 512 MiB guess.
- **the loan's upside is larger than the earlier estimate**: if the chunk's 1 378 MiB came from the cache instead
  of from free VRAM, then free VRAM only ever has to cover the decode side, and the cache could take
  `free_at_load - decode_slack - safety` ≈ 2.3 GB + ~1.7 GB ≈ **4.0 GB (~2 400 slots, +100%)**, with the invariant
  the instrument now makes checkable: **free VRAM stays above ~600 MiB at all times**, chunk or not.
- and the loan's sizing rule is now a measurement, not a formula: `cache = free_at_load - 600 MiB - safety`,
  validated by running the instrument and watching the low water.

That also retires the last open question about Phase A's shape: the borrow does not need to cover "the chunk's
footprint against the planner's estimate" at all. It needs to cover the chunk, full stop, and the free-VRAM
reserve is a property of the decode side that does not change with the chunk's size.

## Step 1 done: free-VRAM growth buys nothing measurable (2026-10-22)

Sweeping the cache with both instruments and `NAIWA_PREFILL_STRICT=1` (`doc`, one 1696-row chunk, 64-token
answers):

| `--expert-cache-gb` | slots | chunk low water | decode low water | turn-1 TPOT | turn-2 TPOT |
|---|---:|---:|---:|---:|---:|
| 2.3 | 27 | 607 MiB | 601 MiB | 50.1 ms | 38.8 ms |
| 2.4 | 28 | 478 MiB | 508 MiB | 49.1 ms | 33.4 ms |
| 2.5 | 29 | 384 MiB | 382 MiB | 48.5 ms | 37.4 ms |
| 2.6 | 30 | 222 MiB | - | `CUDA_ERROR_OUT_OF_MEMORY` | - |

Two things: **+7% slots moves turn-1 TPOT by 3%, inside the run-to-run noise** (turn-3 swings 54-57 ms between
identical configs), so growing the cache out of free VRAM is dead on measurement, not on theory; and the chunk and
decode low waters **track each other** at every cache size (607/601, 478/508, 384/382), which says the low water
is not a fixed requirement but "whatever is left" - until 222 MiB, where something else needs more and the OOM
lands.

**Conclusion: Step 1 closes the free-VRAM avenue.** Everything that is left is the loan, and the loan's value is now
bounded from both ends: it is worth roughly the -10%-per-30%-slots curve (measured), and it requires the chunk to
stop consuming free VRAM entirely (measured).

## The obstacle in the loan's foundation: a slab needs a candle change (2026-10-22)

Designing Phase A ran into the one thing the earlier estimates glossed over. The chunk needs its three projection
stacks **contiguous** (650 MiB total for the largest layer, 217 MiB per projection), and a cache bank is only
~47 MiB, so the borrow range has to be a byte range carved out of a single big allocation. Carving it needs
storages that share one allocation, and in this stack there is no such thing:

- `cudarc` 0.19.10's `CudaSlice` **owns its pointer**: `Drop` calls `free_async`/`free_sync`, and `slice()` on it
  returns a **borrowed `CudaView<'_>`**, not an owned sub-slice. There is no `split` that hands out owned halves.
- candle's `QCudaStorage { data: PaddedCudaSlice { inner: CudaSlice<u8>, len }, dtype, device }` has **private**
  fields and owns an owned `CudaSlice`, so neither crate can fabricate a storage over someone else's bytes.
- CUDA's own answer, VMM (`cuMemCreate` / `cuMemAddressReserve` / `cuMemMap`), exists in `cudarc` **only as raw
  `sys` bindings**, and would bypass candle's allocator entirely.

So Phase A is not "naiwa-only, ~400 lines". Its foundation is one of:

| option | what it takes | cost | risk |
|---|---|---|---|
| **(a) borrowed storage in candle-core** | a `QCudaStorage` variant holding `Arc<CudaSlice<u8>>` + offset + len, with `device_ptr()`/`storage_size_in_bytes()` over it (~40 lines, additive) | small | touches the crate the oracle binary links; additive and inert, but it is a shared dependency |
| **(b) VMM via `sys`** | one physical allocation mapped twice, cache and chunk get separate virtual ranges | large | deep unsafe CUDA, outside candle's allocator and its fences |
| **(c) no slab: alias the chunk's own scratch** | find disjoint-lived temporaries inside a chunk (Strata does this: `bo` reuses `emb`, 320 MiB at 32k rows) | medium, naiwa-only | none for numerics; gain is only the scratch side, so maybe +100-300 MiB of cache |

(a) is the shortest path to the +100% and it is what the design wants; (c) is the only one that stays inside
naiwa. The choice is a scope decision, not a technical unknown, so it is taken deliberately rather than by
drifting: **(a) needs an explicit yes to touching candle-core**, and until then (c) is the only runnable version.

## (a) done: a borrowable storage in candle, and the cache carved from one slab (2026-10-22)

The design's foundation is in, on both sides of the boundary.

**candle (`~/code/candle`, `quantized/cuda.rs`)**: `PaddedCudaSlice` now holds
`inner: Arc<CudaSlice<u8>>` plus `offset`, `len` and `allocated`, and `QCudaStorage::borrow_from(slab, offset,
el_count, dtype, device)` returns a storage over a **range of the slab's allocation** - no copy, no ownership,
and a dtype of its own choosing, because the slab is only bytes. Dropping it frees nothing; the slab's `Arc` is
what keeps the bytes alive. `allocated_bytes()` exposes the padded size so the slab's cursor can advance by
exactly what was handed out. The kernel arguments go through `&data.view()`, which is a `CudaView` over
`offset..offset+allocated` - `cudarc` 0.19 does implement `PushKernelArg` for `&CudaView`, which is what made
this a ~60-line change instead of a rewrite.

**naiwa (`src/moe/slab.rs`, `staging::alloc_bank_in`, `provider::slab`)**: with `NAIWA_CACHE_SLAB=1` the provider
allocates one `--expert-cache-gb` slab per device (as `F32`, i.e. block size 1, so it is a byte pool) and every
layer's bank is a bump-allocated range of it: one range per projection, still one contiguous `[slots, n, k]`
tensor per projection, still a `SlotMemory` per slot. A layer whose bank does not fit what is left falls back to
its own allocation, so an undersized slab costs the lending opportunity and nothing else.

**Both verified:**

- `moe::slab::tests::a_borrowed_range_is_a_view_of_the_slab` (device test, `--ignored`): a range starts at its
  offset, dropping it leaves the slab intact and the address unchanged, and two ranges read as different quantised
  types;
- the **oracle gate is byte-identical in slab mode** (`FRAC=0`, `--no-lookahead`,
  `NAIWA_CACHE_SLAB=1` -> `' Paris. The capital of Japan is Tokyo. The capital of Italy is Rome.'`), and the log
  confirms the path taken: `expert cache: one 2.00 GB slab, banks carved from it`.

The chat corpus is **not** a correctness oracle here: with `--pcie-frac 0.2` (and adaptive swaps) the first token
already differs between two runs of the *same* binary, which M16 recorded. That is why the slab is validated by the
`FRAC=0` gate rather than by diffing chat text.

**What is left of (a)**: the borrow itself. The slab now makes it possible: the low end of the slab is a
contiguous range whose owners are known banks, so a chunk can (i) mark the slots that live there empty, (ii) build
its three projection storages with `borrow_from` over that range at the current layer's dtype, (iii) run, and
(iv) return the range, leaving those slots empty to refill. Everything that needs is now in place except the
sizing (measured: chunk 1 378 MiB, decode low water ~600 MiB) and the fences.

## The borrow itself: implemented and running (2026-10-22)

`NAIWA_CACHE_SLAB=1 NAIWA_PREFILL_BORROW=1` now makes the chunk's layer buffers *ranges of the cache's slab*.
Nothing is copied into them from a buffer of their own; the slots the ranges cover are emptied for the chunk and
come back empty, exactly as after an eviction.

How it fits together:

- `Slab::borrow_at(offset, el_count, dtype)` hands out a range at a fixed offset (0, then each projection's padded
  size), and `Layers::empty_range(base, end)` empties every slot whose bank overlaps it;
- `PrefillStream::borrow_buffers` composes the three projection buffers that way and returns them as the same
  `[SlotMemory; 3]` the ordinary path builds, so `MoEExperts::from_qtensors`, the DMA and the fences are untouched;
- `plan()` counts the layer buffer as **zero** free VRAM when borrowing (and forces one buffer, since every layer
  takes the same range), which is what lets a 2.95 GB cache keep the 8 192-token chunk it used to lose at 2.85.

**Measured on `doc`** (cache 2.3 GB without the borrow vs 2.95 GB with it):

| | slots/layer | chunk footprint | turn-1 TTFT |
|---|---:|---:|---:|
| no borrow, 2.3 GB | 27 | 1 378 MiB | 3.88-4.01 s |
| borrow, 2.95 GB | **35** | **866 MiB** | 3.95 s |

The chunk still costs 866 MiB of free VRAM - that is the activation scratch, which lives in candle's pool and not in
our slab, so it is the part a borrow of *this* kind cannot move. The layer buffer's 650 MiB, which is what made
2.85 GB refuse the chunk, is gone from the budget.

**Two bugs on the way, both the same family as W105:**

1. `empty_range` waited for slot readers *while holding* `layer.state`; the reader was a background fill that
   needed that same lock to finish, so the two spun against each other. Fixed by taking the buffers out under the
   lock, waiting with no lock held, then putting them back.
2. Even then it spun: the fills legitimately hold slot buffers across a chunk. A borrowing chunk must not race a
   fill at all, so `spawn_fill` is now skipped when borrowing - which costs nothing, since
   `NAIWA_NO_PREFILL_FILL` already measured the fills useless during a chunk.

**Open issue, not yet explained**: with `--pcie-frac 0 --prefix-cache-n 0` the borrow run ends in
`invalid CUDA top-1 output` (one turn out of three), while the same config without the borrow cannot even be
compared - its chunk falls back to the CPU (130 s/turn) because 2.95 GB leaves the stream no buffers. So the
borrow's determinism could not be checked against a working baseline in *that* config. The 200-word `FRAC=0`
capture run was byte-identical between modes, but its answer was empty, so that check is weak.

**Directional numbers, not yet a gate** (one run each, `short`; the earlier spreads on this card are ±15%):

| | turn-1 TPOT | turn-2 TPOT | turn-3 TPOT | turn-2 TTFT |
|---|---:|---:|---:|---:|
| no borrow, 2.3 GB | 58.1 ms | 52.9 ms | 36.1 ms | 1.40 s |
| borrow, 2.95 GB | 57.5 ms | 38.7 ms | 32.8 ms | 0.64 s |

The reuse differed between those two runs (0.51 vs 0.76 on turn 2), so this is a direction, not a result: the
mechanism works, and the proper numbers need the 3-round harness against `harness/chat/engines/*.json`.

## The harness gate: the borrow works but is not yet trustworthy (2026-10-22)

Ran the real gate - `harness/chat/run.py --engines naiwa,naiwa-loan --rounds 3`, where `naiwa-loan.json` is
`naiwa.json` with `--expert-cache-gb 2.95` and `NAIWA_CACHE_SLAB=1 NAIWA_PREFILL_BORROW=1`. **It does not pass
yet**: the baseline engine completed every conversation, the loan engine dropped one turn per round with
`no content delta (finish None)` - an empty answer, not an error the server logged. Earlier, in the
`--pcie-frac 0 --prefix-cache-n 0` configuration, the same path failed with `invalid CUDA top-1 output` (one turn
in three), which is what a garbage expert does: NaN in a hidden state makes the router's argmax return a bogus
index.

One real ordering hole was found and fixed on the way: a fill that the previous decode step queued on the copy
stream is still in flight when the chunk begins, and before the borrow it wrote into memory of its own. Now it
lands in the borrowed bytes *after* the chunk's upload and corrupts the layer buffer. `borrow_buffers` now
synchronises the copy stream before it takes the range. That is necessary but not sufficient - the empty answer
survives it, so there is at least one more hole, and the next step is to catch a failing request with the engine's
own log rather than to guess.

**What this means for the milestone**: the mechanism is real and measured (27 -> 35 slots, chunk footprint
1 378 -> 866 MiB, turn-1 TTFT unchanged) but it is **not shippable**, so it stays behind
`NAIWA_CACHE_SLAB` + `NAIWA_PREFILL_BORROW` and out of `engines/naiwa.json`. The slab on its own is different: it
is byte-identical through the oracle gate and inert, so it can stay as the foundation.

**Where to look next**, in the order the evidence points:

1. the borrow empties slots while the decode machinery may still hold state for them (the lookahead channel's
   published batch, `PassCell`'s preload claim, the `--pcie-frac` staging ring's completion event). Fills are
   skipped when borrowing now, but nothing resets those; a quiescent-point check (borrow only when the layer's
   tick/claim state is idle) would make the swap safe by construction;
2. whether the chunk's upload is ordered after *candle's* stream (a decode kernel still reading a slot);
   `CopyStream::after_device_work` exists for exactly that and is not called on this path;
3. and only then the empty-answer case itself, which needs a captured failing request.

## Investigation: two bugs found and fixed, one defect narrowed but open (2026-10-22)

Chased the intermittent failure properly, with an instrument on the chunk's own output
(`NAIWA_BORROW_CHECK=1` reports a non-finite chunk output) and the borrow's per-layer debug line.

**Bug 1 (fixed) - the empty was racing a fill.** `admit` reserves a slot (`expert = Some(id)`,
`buf = None`) *before* the copy and publishes after it. `empty_range` cleared the reservation, the fill
then published, and the slot came back claiming an expert whose bytes the chunk had already
overwritten - exactly the garbage that makes a router return an invalid index. `empty_range` now clears
the slot a *second* time after it has waited for the readers, and the borrow synchronises the copy
stream after the clear so no fill DMA can land on the chunk's buffer.

**Bug 2 (fixed) - the wait was held under a lock**, deadlocking against a fill that needed the same
`layer.state` to finish. Take under the lock, wait without it, put back.

**What is left, and why it is now a different failure.** With both fixes the hard CUDA errors stopped:
the last three reproducer runs each show all 48 layers borrowing (`prefill_borrow: layer 0 ... 47`),
e.g. `layer 0: borrowed 531 MiB, 305 slots emptied`, `layer 47: borrowed 650 MiB, 360 slots emptied`, and
the chunk's own output is finite when checked. What remains is that roughly **one conversation in three
returns no content at all** ("no content delta (finish None)" in the harness) with **no error in the
server log** - the request simply ends. So the chunk is no longer the suspect; what follows it is.

One hard error was captured once before those fixes and is worth keeping because it names a path:
`Model failed with error: unexpected rank, expected: 2, got: 3 ([1, 1696, 2560])`. The only rank-2
consumers here are `Gpu::submit`/`QMatMul` on the decode path, so in that instance the *decode* path was
handed the chunk's 3-D tensor. How it got there is not established: the chunk branch is guarded by
`num_tokens > PREFILL_THRESHOLD` and returns, so something re-entered decode with a chunk-shaped
activation - the MTP verify step and the per-layer `MAX_DECODE_ROWS` ring are the two candidates.

**Verdict**: the borrow mechanism is proven (35 slots/layer, chunk footprint 1 378 -> 866 MiB, turn-1
TTFT unchanged, borrow lines for all 48 layers) but it is **not shippable**, and it stays behind
`NAIWA_CACHE_SLAB=1 NAIWA_PREFILL_BORROW=1`. Two of its bugs are fixed and the third is narrowed to
"something after the chunk ends the turn empty", with the one concrete clue above.

**Next instrument, in order**: (1) dump the first decode step's logits with the borrow on and off for the
same conversation and compare - an empty answer with healthy logits is a sampler/prefix-cache problem, a
degenerate argmax is still a data problem; (2) assert the rank at the decode entry (`xs.dims3()` plus a
check against `MAX_DECODE_ROWS`), which would catch the `[1, 1696, 2560]` case at its source instead of
in `candle`'s matmul.

## The bugfix attempt, and where it stops (2026-10-22)

Chased the remaining defect to a **reproducer** instead of a symptom, and it is a hang, not a wrong answer:

```
prefill_borrow: layer 0: borrowed 531 MiB of the slab, 310 slots emptied
prefill_stream: layer 0 stride 1 dma_wait=781ns compute=55.46ms copy=55.47ms 10.0 GB/s 531 MiB x 3
                                     <- layer 1 never logs a borrow; the request stops here
```

No error is logged, the process stays alive, and the turn returns no content - which is the harness's
"no content delta (finish None)". It reproduces at `--expert-cache-gb 2.3` as well as 2.95, i.e. it is not a
sizing problem, and the baseline engine never shows it.

So `PrefillStream::forward` for layer 1 gets past the decline checks (`stacks` present, `stride` non-zero) and
stops inside `load`, before `borrow_buffers` logs. Three candidates, in order:

1. `buf.ready.wait_on(&self.device)` - the relayout path waits for layer 0's upload completion event;
2. `empty_range` spinning in `wait_for_readers` for layer 1's banks;
3. the first decode-side consumer of an emptied bank (the lookahead channel's published batch or `PassCell`'s
   preload claim) blocking, since layer 0 leaves ~310 slots empty and nothing resets that state.

Fixes applied during this investigation and **kept**, because they are real:

- the fill/empty race (a fill reserved a slot before the clear and published after it, leaving a slot that
  claimed an expert whose bytes the chunk had overwritten) - `empty_range` now clears again after waiting, and
  the borrow syncs the copy stream after the clear;
- the reader wait under `layer.state`, which deadlocked against a fill needing that lock;
- the sizing hole that let the chunk eat the decode reserve (`act * 2 + head` when borrowing, since a borrowed
  chunk still pays activations *and* the step's scratch from free VRAM - measured 1.95x `act`).

**Shipped state**: the borrow is parked. `borrows()` logs a warning that it is experimental and known to hang,
the engine file moved to `harness/chat/engines/naiwa-loan.parked.json` with that status in its `record`, and
`engines/naiwa.json` is untouched. `NAIWA_CACHE_SLAB` stays: it is byte-identical through the oracle gate and
inert, and it is the foundation the borrow needs.

**Next step, precisely**: instrument candidate 1 first - a `dbg_note` on either side of `buf.ready.wait_on` in
`load` names the hang in one run, and it is the only one of the three that is on the chunk path itself.

## Root cause found: the chunk's kernels outlive the borrow (2026-10-22)

The hang/empty-answer turned out to be neither. Instrumenting `PrefillStream::load` (a `dbg_note` on either side
of every wait) showed the chunk completing all 48 layers, and the server then logging, one line after the chunk:

```
ERROR step - Model failed with error: invalid CUDA top-1 output
```

That error comes from the **sampler** (`mistralrs-core/src/sampler.rs:1584`), which bails when the argmax row is
not finite - i.e. the first decode step's logits are NaN. The mechanism:

- the layer buffer is a *range of the cache's slab*, so the cache's slots and the chunk's buffer are the same
  bytes;
- `PrefillStream::forward` returns as soon as its kernels are **enqueued**; `buf.released.record_reads` only
  records them on the fence;
- the next decode step's fills DMA into those same slots in place, and a fill knows nothing about the prefill
  buffer's fence - so it overwrites the layer buffer while the last layer's kernels still read it.

Hence NaN logits, and hence the intermittency: it needs the decode step to start before the chunk's last kernels
retire.

**Fix**: when borrowing, `forward` ends with `copy.after_device_work()` + `copy.synchronize()`, i.e. the chunk is
*finished* before the range is handed back. One sync per chunk, and it makes the aliasing safe by construction
rather than by timing.

**Gate**: `harness/chat/run.py --engines naiwa,naiwa-loan --rounds 3` → **0 failures** (4 classes, 3 rounds,
both engines):

| class | engine | TTFT t1 | TTFT ≥2 p50 | TPOT p50 | decode tok/s | e2e/conv |
|---|---|---:|---:|---:|---:|---:|
| code | naiwa | 1012 ms | 1284 ms | 35 ms | 28.9 | 27.3 s |
| code | **loan** | **963 ms** | **1254 ms** | **32 ms** | **31.5** | **25.1 s** |
| doc | naiwa | 4052 ms | 1049 ms | 43 ms | 23.4 | 29.9 s |
| doc | **loan** | **3751 ms** | **992 ms** | 43 ms | 23.2 | 44.2 s* |
| long | naiwa | 3820 ms | 1224 ms | 48 ms | 21.1 | 46.8 s |
| long | **loan** | 3931 ms | **1128 ms** | **46 ms** | **21.7** | **43.4 s** |
| short | naiwa | 1039 ms | 979 ms | 51 ms | 19.6 | 42.3 s |
| short | **loan** | 1071 ms | **926 ms** | **42 ms** | **23.7** | **34.9 s** |

\* one slow conversation in that class; the per-turn table shows doc turn 1 at 3751 vs 4052 and turns 2-5
faster on every one (`862` vs `1036` at turn 4, `657` vs `859` at turn 5).

So the borrow buys **TPOT 35→32 (code), 51→42 (short), 48→46 (long)** and TTFT ≥2 on every class, which is what
the slot→TPOT curve predicted, at a turn-1 TTFT that does not regress.

**Still open**: the byte-determinism configuration (`--pcie-frac 0 --prefix-cache-n 0`) fails one turn of three
with the same `invalid CUDA top-1 output`, while the same configuration without the borrow does not
(`/tmp/d3-base.log`: 0, `/tmp/d3-loan.log`: 1). So the alias is still not fully fenced in that configuration,
and until it is, the borrow cannot carry a numeric gate - the same rule the slab met (oracle byte-identical) and
this does not yet.

**Status**: `engines/naiwa-loan.json` exists and reproduces the table above; `engines/naiwa.json` is unchanged, so
the default arm still runs the stream. The borrow remains behind both env flags.

## Where the borrow actually stands after the fix (2026-10-23)

The chunk-outliving-its-buffer fix (a synchronise at the end of a borrowed chunk) took the **chat** configuration
from "one turn in three dies" to **3/3 harness rounds with zero failures**, and the borrow is faster there (TPOT
35->32 `code`, 51->42 `short`, 48->46 `long`; TTFT >= 2 better on all four classes). That is the configuration the
milestone measures, and it holds.

The **byte-determinism** configuration (`--pcie-frac 0 --prefix-cache-n 0`) does not hold, and the sweep shows why
it is not simply "the borrow is unfenced":

| cache | borrow | chunk | outcome |
|---|---|---|---|
| 2.3 GB | off | 8192 | 3 turns, no error |
| 2.3 GB | on | 8192 | 3 turns, no error (twice) |
| 2.6 GB | on | 8192 | **hangs**, no error, no chunk log |
| 2.95 GB | on | 5632 | **`unexpected rank, expected: 2, got: 3 ([1, 1725, 2560])`**, deterministically |

So with the borrow the prompt path in that configuration needs roughly the free VRAM it had at 2.3 GB, and both
2.6 and 2.95 fall off a cliff in different ways: a hang at 2.6, and a rank-2 consumer being handed the prompt's
3-D tensor at 2.95. Neither is the aliasing bug fixed above (that one produced NaN logits in the sampler and is
gone from the chat configuration), and neither reproduces at 2.3.

**Honest reading**: the borrow's *mechanism* is sound and its *benefit* is real, but its safe operating point in
the deterministic configuration is "no cache growth at all", which makes it useless there; and the chat
configuration's 3/3 is at a size (2.95 GB) where the deterministic configuration falls over. Until those two
agree, the borrow cannot carry a numeric gate.

**What that leaves**: `engines/naiwa.json` unchanged (the measured arm), `engines/naiwa-loan.json` present as an
experimental arm at 2.95 GB whose numbers are **not** gated, the borrow behind both env flags, and the three
problems to solve written down here: the prompt path's real VRAM requirement in the FRAC=0 configuration, the
rank-2 consumer that receives a 3-D prompt tensor (the one concrete clue, and now reproducible on demand at
2.95 GB + borrow), and the hang at 2.6 GB.

## The numeric gate, done properly: the chunk is bit-exact (2026-10-23)

Earlier attempts at a byte-identity check were worthless for a reason worth writing down: **`--mtp` makes greedy
output non-deterministic** (M16 recorded it - the drafts change which tokens share an expert, and the same prompt
can end differently). Any comparison must drop the drafter. With `--mtp` off, `--pcie-frac 0`,
`--prefix-cache-n 0`, `--seed 0`, same 3-turn `doc` conversation, borrow off vs on (both at 2.3 GB):

| turn | borrow off | borrow on | verdict |
|---|---:|---:|---|
| **1** (the 1725-row chunk) | 363 B | 363 B | **byte-identical** |
| 2 | 337 B | **0 B** | the borrow arm answers nothing |
| 3 | 337 B | 338 B | diverges, as it must after turn 2 |

So **the borrowed chunk is numerically exact**: the same bytes computed through a different address give the same
answer, which is what the design promised (no numeric change, only placement). That is the gate the borrow needed
for its *chunk* half, and it passes.

**What it does not cover**: a later turn that runs another full chunk (no prefix cache means every turn does) comes
back empty on the borrow arm - turn 2 here, and the same symptom the harness sees as "no content delta". So there
is a second defect, on the *second* borrowed chunk of a session, in this configuration. The chat arm (prefix cache
1, so later turns reuse instead of re-chunking) passes 3/3 rounds, which is consistent: it rarely runs a second
chunk.

**Instruments that stay**: `NAIWA_RANK_DBG` (candle prints a backtrace where a tensor's rank is wrong - it did not
fire in the four clean reproducers, so the rank-2 case is rarer than first thought), `NAIWA_BORROW_CHECK` (per-layer
decode finiteness), and the `prefill_load` debug notes.

**Status unchanged where it matters**: `engines/naiwa.json` is the measured arm; `naiwa-loan.json` is experimental
and without a numeric gate; the borrow stays behind both env flags. The next fix has a precise target now - the
second borrowed chunk in a session, with no prefix cache - and a ready-made reproducer (`--mtp` off, FRAC 0,
prefix-cache 0, 3 turns of `doc`).

## The control that settles it (2026-10-23)

Same configuration, four turns, the only difference the borrow:

| arm | stream errors | server log |
|---|---:|---|
| borrow off (2.3 GB) | **0** | clean |
| borrow on (2.3 GB) | **3 of 4 turns** | **clean** |

Each failing turn gets `{"message": "Internal server error."}` on the SSE stream and **nothing** in the engine log -
no `ERROR`, no `WARN`, no `step - Model failed`. So an error is being raised where the API layer swallows it, and
it only happens with the borrow. In this configuration (`--pcie-frac 0 --prefix-cache-n 0`, no drafter) *every*
turn after the first runs a full chunk, so the trigger is the second borrowed chunk of a session - first chunk
byte-identical to the baseline, later ones into a stream error.

That is the precise, reproducible target that is left. Everything else about the borrow is measured and clean:
the chunk is bit-exact, the chat arm (prefix cache 1) passes 3/3 harness rounds and is faster.

**Why this is where the milestone stops**: the remaining defect needs an error path in `mistralrs-core`'s
streaming engine to be made visible (its errors are not logged by default), and then a second-chunk investigation
in a configuration whose only purpose was byte-identity. That is a milestone of its own, and M19 has already
shipped its mechanism - the slab - plus a measured, faster, experimental borrow on top of it.
