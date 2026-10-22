# M18 — the prefill transport: 8 GB/s, one buffer, and what a second one buys

Status: **closed** (2026-10-21): one mechanism, shipped - the shape-changing pool trim is gone (turn-1 doc 4 431 -> 3 882 ms); the cache-for-buffer trade was priced and rejected.
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

**What that changes.** There is no transport headroom left: the copies run at 11.2-13.4 GB/s, i.e. the link's
practical maximum, and `dma_wait` is **~1 us in both modes** (1 buffer: 601 ns-1.07 us; 2 buffers: 611 ns-1.08 us),
so the consumer never waits for the DMA in either. What the second buffer changes is the *width of the overlap
window*: with one buffer the copy for layer L+1 cannot start until layer L's MoE releases the shared buffer, so
it can only overlap layer L+1's attention (~38 ms against a 40 ms copy); with two it runs a whole layer ahead.
That is the measured -16% (4.51 -> 3.78 s), and it is also why a third buffer would buy nothing - the copy is
already hidden - while costing `3 x 650 + 522 + 512 = 2 984 MiB` of free VRAM (cache 2.3 -> ~1.2 GB).

So the transport work is one decision, not a project: **take the second buffer**. After it, turn-1 is
compute (`attention + hc`, 0.46 ms/row), which is a different milestone.

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
2. ~~Close the overlap gap~~ - **not a gap.** `dma_wait` is ~1 us with one buffer and with two, so the DMA is
   hidden in both; the second buffer's -16% comes from widening the overlap window, and a third would add
   nothing (it would cost 2 984 MiB of free VRAM for no measurable change). No further transport work unless
   the layer's own kernels get slower.
3. **Sell the second buffer to the plan.** Decide between cache 1.85 GB (two buffers, -0.73 s per turn-1) and
   2.3 GB (one buffer, more slots) with a 3-round corpus, since M10 measured the cache as TPOT-neutral there.
   Gate: `doc` turn-1 <= 3.8 s at no `TTFT>=2` or TPOT regression.
4. **Read fewer bytes at large chunks.** Only reachable through selection (read what the cache already holds),
   whose ceiling at 2.3 GB is ~10% of one layer pass - M17's finding. Keep it as a reason to prefer item 3's
   cache over item 4's selection, not the other way round.

## Pricing the second buffer: it loses (measured, 2026-10-21)

The trade was a config change (`--expert-cache-gb 2.3 -> 1.85`), 3 rounds each, everything else identical:

| class | turn-1 (1 buffer) | turn-1 (2 buffers) | TTFT>=2 | TPOT | reuse |
|---|---:|---:|---:|---:|---:|
| doc | 4 431 ms | **3 654 ms (-18%)** | 1 153 -> 1 111 | **44.9 -> 51.0** | 0.98 |
| long | 4 286 ms | **3 402 ms (-21%)** | 1 062 -> 1 148 | **45.8 -> 49.2** | 0.98 |
| short | 1 065 ms | 1 021 ms | 1 185 -> 1 052 | 46 -> 45 | 0.89 -> 0.91 |

The turn-1 gain is real and matches the probe (-16..-21%). The TPOT cost is also real: medians over 18 `doc` turns
(39.8-49.6 IQR vs 45.0-54.2) with the **acceptance rate unchanged** (50.0% vs 49.2%), so it is the decode expert
cache losing 231 slots (1 206 -> 975), not the drafter.

Per conversation that is ~0.8 s saved on turn 1 against ~4 s lost on ~650 generated tokens at +6 ms. **Reverted
to 2.3 GB / one buffer**: at this cache size the 450 MiB is worth more to decode than a second buffer is worth
to prefill.

Two follow-ups that do *not* work, both measured:

- lowering `--max-num-batched-tokens`/`--max-prefill-chunk-tokens` 8 192 -> 4 096 shrinks the *requirement*
  (act 522 -> 426 MiB) but not the *free* VRAM, and the plan still sees 1 983 MiB against 2 291 needed;
- so the only way to have both buffer and cache is to free real VRAM, and the only lever that size is
  `--max-seqs` (115 MiB per slot): 4 -> 1 plus the smaller chunk would fit, at the cost of serving one
  sequence at a time. That is a concurrency decision, not a free lunch, and it is out of scope here.

## The shape-change trim was 40% of a chunk (found and removed, 2026-10-21)

While answering "how should the prefill claim be measured", the stream's per-layer probe showed **35 layer-shape
changes in one 1 725-token chunk**, each costing **42-47 ms** in the `!fits` path: free the old buffer, call
`trim_device_pool` (a context synchronise plus a pool trim), allocate the next shape's. That is **~1.5 s of a
3.7 s chunk**, and it is pure overhead: the model's GGUF mixes quants per layer, so it has seven layer shapes.

`trim_device_pool` is now **off** in that path (`NAIWA_SHAPE_TRIM=1` restores it for an A/B). The re-allocation
becomes a pool hit - **median 31 us**, min 1 ms - with no OOM and the plan's free VRAM unchanged (1 983 MiB at
cache 2.3, 2 559 MiB at 1.8), i.e. the pool does not hoard the way the original comment feared.

| measurement | with trim | without trim |
|---|---:|---:|
| re-allocation, median | 43 ms | **31 us** |
| `doc` turn-1, two buffers | 3.71 s | **3.24 s** |
| `doc` turn-1, one buffer | 4.51 s | **4.05 s** |

3 rounds, medians, shipped config (cache 2.3, one buffer):

| class | TTFT t1 | TTFT >= 2 | TPOT | reuse | e2e | before |
|---|---:|---:|---:|---:|---:|---|
| doc | **3 882 ms** | 1 132 ms | **42 ms** | 0.98 | **31.4 s** | 4 431 / 1 153 / 45 / 34.4 |
| long | **3 762 ms** | 1 145 ms | 45 ms | 0.97 | **41.0 s** | 4 286 / 1 062 / 46 / 52.9* |
| short | 1 034 ms | **921 ms** | 46 ms | 0.89 | **34.0 s** | 1 065 / 1 185 / 46 / 38.3* |

So turn-1 is -12% on `doc` and `long` for a two-line change, with TPOT unchanged and the oracle byte-identical
(both variants) and 42/42 tests.

## Gates

- Oracle unchanged (`frac 0 + --no-lookahead`, with and without `--prefill-stream`), 36/36 tests.
- Medians of >= 3 turns; `TTFT`/`TPOT`/`reuse` are the comparable columns; `e2e` is not when generated lengths
  differ.
- Every item reports its own chunk-level number (GB/s, copies/layer, per-layer ms), not only a class median.
