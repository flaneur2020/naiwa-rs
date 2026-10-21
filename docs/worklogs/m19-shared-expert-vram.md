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

**P1 — the arena, no behaviour change.** Allocate the cache, the borrow region, the staging ring and the
drafter pool from one arena; log the partition at startup; `plan` consumes the borrow region's size instead of
`free`. Gate: `MISTRALRS_VRAM_DUMP` shows the same tensors, oracle byte-identical (both variants), 36/36 tests,
3-round corpus within noise of M18's baseline (4 431 / 1 153 / 45 on `doc`).

**P2 — spend the headroom on buffers.** With the borrow region no longer competing with `free`, the 512 MiB
"headroom" floor can drop to a measured slack (~128 MiB, the amount that actually OOMed in M10), and the
region can hold **two** layer buffers plus activations: `2 x 531 + 522 + 65 + 128 = 1 777 MiB`, which the
current `free` (1 983 MiB) funds at startup. Gate: `plan` grants 2 buffers with `--expert-cache-gb 2.3`
untouched, `doc` turn-1 <= 3.7 s **and** TPOT <= 46 ms - i.e. the M18 trade undone.

**P3 — lend the region back to the cache.** Between prefills the region becomes `531 MiB / 1.95 MiB = 272`
extra slots for its shape's layer (a "loan" flag makes the LRU and the lookahead fill treat them as normal
slots; the prefill stream takes them back with a fence, and they are refilled afterwards like strata does).
Gate: cache slots +~270 (1 208 -> ~1 480, +22%) with TPOT <= 45 ms, and the refill cost measured per turn
(expect ~50-100 ms of DMA on the first decode step after a prefill) - if the refill eats the gain, keep P2 and
stop.

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

## Gates

- Oracle `frac 0 + --no-lookahead` byte-identical, with and without `--prefill-stream`; 36/36 tests.
- 3 rounds, medians; `TTFT`/`TPOT`/`reuse` comparable, `e2e` not when generated lengths differ.
- The partition must be visible in one startup line (`cache N slots | borrow M MiB | ring K slots | drafter D
  MiB`) so a run's own log explains its VRAM, the way M18's plan line does today.
