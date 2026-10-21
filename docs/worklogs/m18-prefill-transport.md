# M18 — the prefill transport: 8 GB/s, one buffer, and what a second one buys

Status: **not started** (opened 2026-10-21).
Parent: [m17-prefill-cost.md](m17-prefill-cost.md).
One mechanism this milestone: **the prefill stream's own transport** — the copy rate, the buffer count, and
which bytes are chosen — as opposed to M17's question of how many bytes a chunk must read at all.

## The measurements this starts from

| observation | value | how |
|---|---:|---|
| turn-1 prefill, one 650 MiB buffer (`--expert-cache-gb 2.3`) | **4.44 / 4.58 / 4.51 s** | `bin-b1`, `bin-ple`, 1 725-row `doc` turn |
| turn-1 prefill, two buffers (`--expert-cache-gb 1.8`) | **3.73 / 3.84 / 3.76 s** | `bin-sdbg`, `bin-tail4`, `bin-e2b`, same turn |
| second buffer | **-0.73 s (-16%)** | difference of the two rows |
| implied copy rate when the DMA is the constraint | **~8 GB/s** (23 GiB / 3.18 s) | a 150-new-row turn, two buffers, `NAIWA_STREAM_DBG` |
| `dma_wait` per layer with two buffers | **~600 ns** | `NAIWA_STREAM_DBG`, 1 725-row chunk |
| streamed MoE compute per layer | 0.5-11 ms | same probe |
| PCIe ceiling on this box | 13.4 GB/s | strata's own probe, quoted in M10 |

So the two numbers disagree in an interesting way: with two buffers the per-layer *wait* is 600 ns, i.e. the
DMA is fully hidden, yet a chunk whose DMA cannot be hidden (150 new rows, little attention to overlap with)
runs at 8 GB/s, not 13.4. **A second buffer buys overlap; it does not buy rate.**

## The probe, built: the copies are already at line rate (2026-10-21)

`NAIWA_STREAM_DBG` now prints a per-layer transport line (`copy`, `GB/s`, `MiB x copies`), and a new
`NAIWA_STREAM_SYNC=1` makes the upload blocking so the copy's own wall time is visible rather than the
queue-side latency (`CopyDone::wait_on` only *enqueues* a stream wait, it does not block the host - the first
version of this probe measured 6-29 GB/s at random because of that).

Serial copies of one layer (481 MiB in 3 blobs): **92-109 ms, i.e. 4.6-5.5 GB/s**. That looked like a 2.5x
transport loss, so the box itself was calibrated (`cargo test h2d_bandwidth`, 512 MiB copies):

| case | rate |
|---|---:|
| pageable host memory | 10.6 GB/s |
| `cuMemHostRegister` (what `pin_stack` does) | **13.4 GB/s** |
| 3 blobs of 170 MiB instead of one 512 MiB blob | 13.4 GB/s |
| after `advise_huge_pages` | 13.4 GB/s |
| on a second stream, built like `CopyStream` | 13.4 GB/s |

So the link, the blob shape, the huge-page advice and the separate copy stream are all fine: a layer's copy
takes **40 ms at 13.4 GB/s**, and the remaining ~55 ms of the serial measurement is the **fence wait** that
serial mode forces (the previous readers' event). The "8 GB/s implied rate" in M17 was a modelling artifact:
chunk time is not copy time.

**What that changes.** There is no transport headroom left - the copies already run at the link's practical
maximum. The turn-1 floor is *overlap*: per layer with two buffers we measure ~78 ms of which the copy is
40 ms, so an ideal pipeline would be `max(40, attention+MoE)`; the sum being 78 ms means roughly half the copy
still is not covered. That is what a buffer count buys, and it is why the second buffer measured -16%
(4.51 -> 3.78 s) while a third one is expensive: `3 x 650 + 522 + 512 = 2 984 MiB` of free VRAM, i.e. the
expert cache would drop from 2.3 GB to ~1.2 GB.

## Why 8 GB/s and not 13.4 (superseded by the measurement above)

Candidates, cheapest to check first:

1. **Copy shape.** A layer is copied as three whole-stack blobs (gate, up, down), `256 x 1.95 MiB / 3` each —
   i.e. ~168 MB per `cudaMemcpyAsync` from pinned host memory, 144 copies per chunk. That shape should reach
   the link's rate, so this is a hypothesis to falsify, not the leading one.
2. **Event discipline.** Every layer's copy waits on the buffer's `released` fence (this layer's readers) and
   the consumer waits on `ready`. With one buffer those two fences serialise DMA and compute; with two they
   interleave, which is exactly the -16% above. If the fences are *also* costing more than they should (one
   stream wait per layer, `wait_for_readers` spinning), the rate would look like 8 GB/s even when the copy
   engine is idle half the time.
3. **Host side.** The stacks are pinned by `pin_expert_stacks`; a layer that failed to pin never attaches a
   stream at all, so this should not apply — but the pin is per *layer*, and a partially pinned stack would
   still copy through pageable memory.

The first thing to build is therefore a **transport probe**: inside `load`, count bytes and measure the copy's
own wall time (queue -> `ready` completion) per layer, and report `GB/s`, `copies/layer`, `bytes/copy`, plus
how long `wait_for_readers` and `ready.wait_on` each take. That separates "the link is slow" from "we are not
keeping it busy".

## What a second buffer is worth, in the config we ship

It is already implemented and already measured (-16%). It is *not* what we run: the chat config keeps
`--expert-cache-gb 2.3`, and `prefill_stream::plan` needs

```
free >= stride x 650 MiB + activations(8192) + activations/8 + max(512 MiB, free/8)
     >= 1300 + 522 + 65 + 512 = 2399 MiB
```

while that config leaves 1 984 MiB. Buying the second buffer therefore costs **~415 MiB of expert cache**
(2.3 -> ~1.85 GB), which M10 measured as TPOT-neutral at chat, so the trade is: -0.73 s on every turn's prefill
against ~10% fewer cache slots (a tail-hit difference of a few percent, per M17).

## Work items

1. ~~Transport probe~~ - **built, and it retired the premise.** The copies are at 13.4 GB/s (the link's practical
   rate, calibrated with `cargo test h2d_bandwidth` / `h2d_three_parts` / `h2d_second_stream`); the "8 GB/s"
   was chunk time divided by bytes. `NAIWA_STREAM_SYNC=1` remains as the instrument for copy-vs-fence time.
2. **Close the overlap gap instead of raising the rate.** Per layer with two buffers: ~78 ms total, 40 ms copy,
   so the ideal is `max(40, rest)`. Options, cheapest first: (a) prefetch *deeper* than one layer (a third
   buffer costs 2 984 MiB of free VRAM, i.e. cache 2.3 -> ~1.2 GB); (b) split the layer's copy so its second
   half starts after the first half's fence instead of after all three blobs; (c) check that the fence is
   recorded as early as it could be (it is recorded after the MoE, not after the last kernel that reads the
   buffer). Gate: `doc` turn-1 <= 2.6 s (from 3.78 s with two buffers) with the oracle unchanged.
3. **Sell the second buffer to the plan.** Decide between cache 1.85 GB (two buffers, -0.73 s per turn-1) and
   2.3 GB (one buffer, more slots) with a 3-round corpus, since M10 measured the cache as TPOT-neutral there.
   Gate: `doc` turn-1 <= 3.8 s at no `TTFT>=2` or TPOT regression.
4. **Read fewer bytes at large chunks.** Only reachable through selection (read what the cache already holds),
   whose ceiling at 2.3 GB is ~10% of one layer pass - M17's finding. Keep it as a reason to prefer item 3's
   cache over item 4's selection, not the other way round.

## Gates

- Oracle unchanged (`frac 0 + --no-lookahead`, with and without `--prefill-stream`), 36/36 tests.
- Medians of >= 3 turns; `TTFT`/`TPOT`/`reuse` are the comparable columns; `e2e` is not when generated lengths
  differ.
- Every item reports its own chunk-level number (GB/s, copies/layer, per-layer ms), not only a class median.
