# M23 - TTFT (prefill)

**Status:** open, **first measurement done, first lever withdrawn**. The measured curve stands
(`TTFT = 1.15 s per chunk + 1.67 ms per token`, §6); the routed upload built in §5 was **rolled back** after it
measured -4%: a prefill chunk touches essentially every expert (§6), which is what prefill does by construction.
**Parent:** M22 (TPOT line closed at doc -6.4%), M21 (the continued-turn chunk cost model).
**Why now:** TPOT is at its floor. TTFT is the largest remaining term: doc/short spend **6.8-6.9 s per conversation**
in prefill, 7-20x the remaining TPOT headroom.

## 1. Where a 1 725-token turn-1 actually goes

One request, `doc` turn 1, `--mtp-n-predict 3`, `NAIWA_STEP_BUDGET=1`:

```
prefill tokens=1725  ttft=3.87s  => 2.24 ms/token      (engine reports 345 T/s prefill)
```

The MoE steps inside it - and there are only **three** `STEP_BUDGET` lines for the whole chunk:

| step | rows implied | total | expert_dev | `plan` (doorbell wait = device) | pairs | distinct |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 63.5 ms | 20.9 | 53.9 | 480 | 480 |
| 2 | **1 392** | **416.2 ms** | 142.4 | 399.7 | **13 920** | **4 391** |
| 3 | 1 | 28.7 ms | 18.0 | 21.3 | 480 | 479 |
| | | **508 ms** | 181 | | | |

So **87% of TTFT is outside the instrumented MoE step**: 3.87 s of TTFT against 0.51 s of step time. The expert kernels
are 0.18 s of it (4.7%).

## 2. (withdrawn - see §5/§6) The missing time is PCIe, and the arithmetic is exact

`--prefill-stream` keeps **two device buffers of one whole layer of experts each** and uploads a layer per chunk step,
verbatim, from the pinned host stacks (`prefill_stream.rs` module docs). The startup line reports the size:

```
prefill (first chunk): chunk 2048, 1 x 650 MiB layer buffer(s) + 378 MiB activations + 512 MiB headroom = 1540 MiB
```

48 layers x 650 MiB = **30.5 GB per chunk**. At the H2D rate calibrated in M20 (**13.4 GB/s**) that is **2.3 s** -
which is the bulk of the unexplained 3.36 s, with the rest being the unhidden first layer plus allocation.

**Prefill is a transfer, not a computation.**

## 3. (withdrawn - the 4 391 counts cache slots, not experts) it moves 2.8x more bytes than the chunk reads

Step 2's own counters answer this: **13 920 pairs, 4 391 distinct experts** across 48 layers, i.e. **91 distinct
experts per layer out of 256 = 35.7%**. The upload is not routed at all - it streams every expert of every layer.

| | today | if only the routed experts were uploaded |
|---|---:|---:|
| bytes per chunk | 30.5 GB | **10.9 GB** |
| at 13.4 GB/s | 2.3 s | **0.81 s** |
| projected TTFT | 3.87 s | **~1.6-2.0 s** |

Note the same counter also shows prefill's `(row, expert)` redundancy is much worse than decode's: **68% grouping
headroom at 1 392 rows vs 27-34% at 4** - which is why routing the upload pays here even though it does not for decode
(M22.4 §2 measured grouping at 1% for the GEMV).

## 4. (withdrawn with §5) The mechanism to build

Per layer, the router runs before that layer's experts, so the exact needed set *is* known before the upload - just
not early enough to hide it behind the previous layer. That trade is worth taking, because the upload is 2.8x smaller
than the pipeline it would give up:

| schedule | per layer | per chunk (48) |
|---|---:|---:|
| today: upload 650 MiB, pipelined | 48 ms upload (hidden behind 3 ms compute: **not** hidden) | 2.3 s |
| **gather 232 MiB, run unpipelined** | 17 ms upload + 3 ms compute | **0.96 s** |
| ideal: gather 232 MiB, pipelined | 17 ms (hidden) | 0.81 s |

The pipelining today is already worthless for that reason: 48 ms of upload against 3 ms of compute cannot overlap into
anything. So the first build is the simple one - **route the upload** (gather the distinct experts of the layer into
the stream buffer, then run), with the buffer sized for the worst case the router can produce.

### Gates for M23

1. `doc`/`short` turn-1 TTFT: **3.87 s -> <=2.2 s** (median of 3, no failure retries), oracle arm byte-identical.
2. Prefill T/s in the engine line: **345 -> >=600**.
3. Never worse on `long`/`code` (their chunk sizes differ).

### Refutations carried in from M21

`--prefill-stream-min-tokens` 64/32/16/8/1 all equivalent; `MISTRALRS_MERGE_SUFFIX_BELOW` 2-4x slower;
`NAIWA_PREFILL_THRESHOLD` 32/64/128/256 all equivalent; chunk 1024 costs +1.9 s. So the chunk *boundaries* are not the
lever - the bytes are.

## 5. Built the routed upload - falsified, and **removed** from the tree

Implemented `NAIWA_PREFILL_ROUTED=1`: per layer, the distinct experts this chunk routes to are gathered out of the
host stack and copied into their own places in the full-size layer buffer (`upload_experts_async`), so the *layout*
and every kernel stay identical - only fewer bytes move. The layer-`l+1` prefetch is dropped with it (§4).

| arm | bytes/layer | TTFT (1 725-token doc turn 1) |
|---|---:|---:|
| whole layer (baseline) | 650 MiB | 3.87 s |
| **routed** | **487 MiB (-25%)** | **3.72 s (-4%)** |
| | | |

The saving is 25% and not the 64% §3 predicted, and the direct instrumentation says why:

```
STREAM_DBG prefill_stream: layer 0 stride 1 dma_wait=901ns compute=51.8ms copy=53.0ms 9.6 GB/s 487 MiB x 705
STREAM_DBG prefill_stream: layer 47 stride 1 dma_wait=691ns compute=61.4µs copy=1.5ms 412.6 GB/s 594 MiB x 702
```

`x 705` copies = **235 of 256 experts** per layer. A 1 392-row chunk makes 13 920 draws over 256 experts - **54 draws
per expert** - so it uses essentially all of them, and routing can only trim the tail. **§3's "35.7% of the layer" was
wrong**: that 4 391 came from the `pairs/distinct` counter, which counts *cache slots* for hits, not expert ids.

### And the upload is not where the time is

Three measurements, same chunk:

| term | value | share of 3.87 s |
|---|---:|---:|
| MoE steps (budget instrument, all 3 steps) | 508 ms | 13% |
| expert compute inside the stream (48 layers, summed) | 155 ms | 4% |
| layer-buffer reallocations (37 shape changes, trim off) | 3 ms | **0.1%** |
| **everything else - the dense path** | **~3.3 s** | **~87%** |

So prefill is **not** transfer-bound. The 650 MiB upload costs at most a few hundred ms, and cutting it by a quarter
moved TTFT by 4%. The 87% is in the non-MoE half of the layer: attention (this model's linear/DeltaNet layers),
the router, the shared expert and the norms - **2.1 ms per row**, against ~0.11 ms per row for the experts.

That is now the M23 target, and it is a *different* kind of work: a per-stage timer inside a prefill layer, then the
stage that owns the 2.1 ms/row. The chunk-boundary levers were already refuted in M21, and the transfer levers are
refuted here.

**Removed.** `NAIWA_PREFILL_ROUTED`, `copy_stream::upload_experts_async` and the per-layer gather are back out of the
tree: prefill routes to (nearly) all experts by construction - `topk = 10` over 256 experts means a few dozen rows
already cover ~78% of a layer's experts (§6) - so "only upload what is routed" is not a mechanism, it is a rounding
error. Reverted in full, no traces, no gated leftovers.

## 6. The TTFT curve: a 1.15 s per-chunk constant, and its exact explanation

Four cold requests, same server, same configuration (`--expert-cache-gb 2.2`, stream on, routed off):

| prompt tokens | TTFT | ms/token |
|---:|---:|---:|
| 52 | 1.24 s | 23.9 |
| 130 | 2.20 s | 16.9 |
| 325 | 2.31 s | 7.09 |
| 871 | 2.61 s | 3.00 |

Fit: **`TTFT(N) = 1.15 s + 1.67 ms x N`**. The constant, not the slope, is what a chat conversation pays: `doc`'s four
turns spend **~4.6 s of their 6.8 s** on it.

### Where the constant comes from - and the arithmetic is exact

A chunk streams every layer's expert stacks once: **23.4 GiB of expert weights** (the model's whole expert set) at the
~20 GB/s that pinned host memory sustains here = **1.17 s**. Measured: 1.15 s. The constant *is* the weight stream.

And it cannot be routed away, which is why §5 moved only 4%: with `topk = 10` over 256 experts, one row touches 3.9% of
a layer's experts, so a chunk of `R` rows touches `1 - (1 - 10/256)^R`: **40 rows -> 78%**, 1 392 rows -> 100%. The
`235/256` the instrument reported is not an accident of a big chunk; it is what a few dozen rows already imply.

| chunk rows | experts touched per layer | cheapest possible stream |
|---:|---:|---:|
| 40 (a continued turn) | ~200 / 256 (78%) | 18 GiB |
| 1 392 (a fresh document) | 256 / 256 (100%) | 23.4 GiB |

So the floor for one chunk on this card is **~1.2 s**, and `doc`'s four turns cannot go below ~4.8 s of TTFT without
keeping the experts resident - which needs 23.4 GiB of a 12 GiB card, i.e. a different machine, not a different kernel.

### What M23 therefore delivers, and what it cannot

| lever | effect | status |
|---|---|---|
| routed upload (§5) | 650 -> 487 MiB/layer, TTFT -4% | **built, measured, removed**: prefill touches every expert |
| bigger chunks (amortise the constant) | the constant is per chunk, so this is the only lever with real leverage | chunk is already at the VRAM ceiling (2 048 rows, M20/M21) |
| keep experts resident | would remove the constant entirely | 23.4 GiB in 12 GiB: impossible here |
| cache-only prefill (rings / `--prefill-stream-min-tokens`) | shorter chunks avoid the stream | refuted in M21, all settings equivalent |
| attention / dense path | 87% of *compute*, but compute is the 1.67 ms/token slope, not the constant | measured: MoE steps are 508 ms of a 3.87 s chunk |

**M23's honest result: TTFT is 1.15 s of unavoidable weight streaming per chunk plus 1.67 ms/token of compute - a
prefill chunk needs essentially every expert, so no amount of routing, gathering or chunk-shaping moves the constant;
only residence would, and 23.4 GiB does not fit in 12 GiB.** Closing the remaining gap needs residency, not
tuning - which reframes the *conversation* as the unit: fewer chunks per conversation is worth 1.2 s each, and that is
where the next milestone should look (`--prefix-cache-n`, continued-turn chunking, M21's `chunks x 600 ms` law).

## 7. Instrument attempt: the prefill stage profile has to move into `cuda/gdn.rs`

Tried to attribute the 1.67 ms/token slope with a per-stage timer (`NAIWA_LAYER_PROFILE=1`): a guard in
`models/qwen3_next.rs::Attention::forward`, a conv/recurrence split in `gdn/layer.rs::forward_recurrent_core`, and
router/experts/shared timers in `SparseMoeBlock::forward`. **None of the three ever fired**, and that is itself the
answer to where the code is:

- the sparse MoE is not the fork's `SparseMoeBlock` - naiwa-rs's `moe::cached_layer` owns that forward, so nothing in
  `qwen3_next.rs` runs for our layers' experts;
- the dense `Attention::forward` in the same file is not reached either: this model's layers are GDN (gated DeltaNet),
  and its prefill goes through `try_forward_grouped_packed_gdn` -> `cuda/gdn.rs`, the grouped/packed kernel path, with
  `gdn/layer.rs` only as the wrapper.

So a stage profile for prefill has to instrument `mistralrs-core/src/cuda/gdn.rs` (and the paged-attention prefill for
the few full-attention layers), not the model file. The fork is back to pristine for those two files: an instrument
that never fires is worse than none, and the numbers it was meant to refine are already in §6.
