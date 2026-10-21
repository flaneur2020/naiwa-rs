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

## Gates

- Oracle `frac 0 + --no-lookahead` byte-identical, with and without `--prefill-stream`; 36/36 tests.
- 3 rounds, medians; `TTFT`/`TPOT`/`reuse` comparable, `e2e` not when generated lengths differ.
- The partition must be visible in one startup line (`cache N slots | borrow M MiB | ring K slots | drafter D
  MiB`) so a run's own log explains its VRAM, the way M18's plan line does today.
