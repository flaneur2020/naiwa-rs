# M17 — what a prompt chunk costs: expert saturation and the per-row tax

Status: **not started** (opened 2026-10-21 to hold the measurement that M16 ended on).
Parent: [m16-mtp-reuse.md](m16-mtp-reuse.md).
One mechanism this milestone: **stop paying for experts a chunk does not need, and stop paying a host
round trip per prompt row**.

## Where M16 left the chat numbers

3 rounds, `short`+`doc`+`long`, naiwa only, medians, `--expert-cache-gb 2.3`, MTP + resident drafter +
`MISTRALRS_ALIGN_CHECKPOINTS=1`:

| class | TTFT t1 | TTFT ≥2 | TPOT | reuse | strata |
|---|---:|---:|---:|---:|---|
| doc | 4 431 ms | **1 153 ms** | 45 ms | 0.98 | 2 430 / 631 / 39 / 0.99 |
| long | 4 286 ms | **1 062 ms** | 46 ms | 0.98 | 2 273 / 1 018 / 39 / 0.99 |
| short | 1 065 ms | 1 185 ms | 46 ms | 0.89 | 1 035 / 625 / 39 / 0.95 |

Reuse and `long`'s TTFT≥2 are at parity. What is left is prefill: `doc`/`long` turn 1 (4.3 s against 2.4 s)
and the ~1.1 s that remains on every turn's prefill.

## The two-term model (measured, not estimated)

Two chunks on the same box, cache 1.8 GB, two stream buffers, `NAIWA_STREAM_DBG` showing `dma_wait` of
**~600 ns** per layer (the second buffer hides the DMA) and 0.5–11 ms of streamed MoE compute:

| chunk | measured | fit |
|---|---:|---|
| 1 725 rows | 3.73 s | 2.93 + 0.46 x 1.725 = 3.73 s |
| 150 rows | 3.00 s | 2.93 + 0.46 x 0.150 = 3.00 s |

so

```
chunk ≈ 2.93 s (fixed) + 0.46 ms x rows
```

### Term 1 — expert saturation: 23 GiB per chunk, whatever the row count

Routing is `topk = 10` out of **256 experts per layer** (`qwen4exp.expert_count = 256`). A chunk of *r* rows
produces `10 r` lookups, so from ~26 rows upward the distinct set is the whole layer. One layer's experts are
`256 x 1.95 MiB = 500 MiB`, and there are 48 of them: **23 GiB per chunk, independent of the row count** —
which is why a 150-row turn prefill costs the same 2.93 s as a 1 725-row one, and why raising
`PREFILL_THRESHOLD` 32 -> 256 did nothing (measured: 4.57/4.11/3.83 s vs 4.59/4.08/3.86 s; the miss side then
pays the same 23 GiB on the CPU at 10 GB/s instead of PCIe at 13.4).

**The ≤31-row tail is the same thing one size down.** Every prompt is prefilled as `[0, align_down(n, 32))`
plus a remainder of at most 31 rows, and that remainder now takes the decode split (M16 item 6) — 31 rows x 10
lookups still spans ~150-256 distinct experts per layer, so the tail costs **0.81 s**, not 0.01 s. It is the
price of putting the recurrent checkpoint on a block boundary; the alternative would be snapshotting
mid-forward, which the paged KV + GDN layout cannot do.

### Term 2 — the per-row tax: 0.46 ms/row = 9.7 us/row/layer

Everything in a layer that is not the routed experts, and it scales with rows:

| piece | note |
|---|---|
| QSA attention | sparse selection over the KV up to the row's position |
| **PLE gather** | the per-layer-position embedding table is **28 GB** and does not fit (`ple: Qwen4-Exp PLE table does not fit in device memory; rows are gathered on the host every step (slow) available_gb=4 needed_gb=28`), so every row is gathered **on the host**, every layer |
| `hc_*` hyper-connections | two BF16 dense passes per layer (`[10240, 320]`-shaped) |

That is 0.80 s of this milestone's 4.4 s `doc` turn-1 (1 725 rows), and 0.07 s of a 150-row turn — so turn 1
and long prompts only.

## Why strata does not pay either term

- **No tail, no fixed term on a new turn**: its per-session KV is a *dense* buffer (`QsaState`), so
  `conversation_kv_save/restore(upto)` snapshots at an arbitrary length — the turn's exact end. Ours is a
  paged, hash-keyed cache that requires block-aligned restore lengths, which is why M16 needed the
  commit-to-boundary trick and still leaves a ≤31-row tail.
- **A 28 GB PLE table with a lookup table on the SSD**, prefetched, instead of a per-row host gather.

## The tail: what the reference does, and what we measured (2026-10-21)

strata's prefill is chunked the same way (`Prefill`: "the prompt's positions [pos0, pos0 + n) are processed in
chunks of `chunk` tokens through all 48 layers"), but it has **no alignment tail**: its per-session KV is dense,
so `conversation_kv_save/restore(upto)` snapshots the turn's exact end. Its last chunk is an ordinary remainder
(`n % chunk`, hundreds of rows at `chunk` = 1,024), not a <=31-row fragment.

For small chunks it switches mechanism at a measured threshold:

| chunk | path |
|---|---|
| >= `stream_all_min()` = **1024** tokens | "every non-resident expert of every layer streams in a fixed order through a `ring_slots()`-slot ring (**nearly all 512 are routed at such a chunk**)" |
| < 1024 (and the last chunk) | MMQ + an **8-slot `STAGE` ring**, experts grouped by expert (`mmq_resident_sort_rows`): **resident ones read from the VRAM tier**, the rest streamed; buffers sized for `stream_all_min() - 1` |
| optional | `STRATA_PREFILL_CPU_SHARE` hands the CPU pool the non-resident experts few tokens route to - "**-20..-35% at 512 and 1000 tokens**", on by default below 1,024 |

So the shape is "one expert-supply mechanism, two selection granularities": whole-layer for big chunks, routed
for small ones. Ours is currently split the other way: the prefill stream (big) is whole-layer but **cache-blind**,
and the decode split (small) is cache-aware but sends its misses to the CPU.

### Three measurements that changed the plan

All on the same 3-turn `doc` conversation, `--expert-cache-gb 1.8`, alignment on (E2):

| `--pcie-frac` | turn 2 TTFT | turn 3 TTFT | staged / cpu |
|---|---:|---:|---|
| 0.0 (all CPU) | 2.24 s | 1.61 s | 0 / 38.7% |
| 0.2 (today) | 1.38 s | 1.18 s | 14.1 / 28.8% |
| 1.0 | **1.23 s** | **1.12 s** | 36.4 / 4.3% |

so staging more of a small chunk's misses *does* help it (my RAM-bandwidth argument that the CPU path must
win was wrong) - but the effect is small, and it is not free:

| ring slots (`NAIWA_STAGING_SLOTS`) | turn 2 | turn 3 |
|---|---:|---:|
| 16 (default) | 1.31 s | 2.56 s |
| 128 | 1.22 s | **5.49 s** |
| 512 | - | server would not start (1 062 MiB of ring + cache + buffers) |

A deeper ring drags 8x more DMA per layer (256 MiB x 48 per step) and starves the next steps; 512 slots does
not fit. And `--pcie-frac 1.0` over a full 3-round corpus is a net loss: `TTFT>=2` 1 153 -> 1 023 ms but
**TPOT 45 -> 56-61 ms**, because every decode step now stages more misses too.

**Conclusion: the tail is not a staging problem.** At ~50 rows it needs ~all 256 experts of every layer
(50 x 10 = 500 lookups), i.e. the 23 GiB pass again, and our existing split - a 16-slot ring staging the
high-weight few, the CPU pool taking the bulk at ~27 GB/s from RAM - is already the better operating point
(strata's own CPU-share default says the same thing). What is left is to **make the tail smaller**, which is
what E2 already did (restore point at the previous turn's last block), or to stop needing a tail at all,
which is a KV-layout question (dense per-session KV), not a scheduling one.

## Work items

1. **Let the big chunk warm the expert cache, so the tail hits.** The prefill stream DMAs a layer's experts
   into a scratch buffer and never tells the cache; the tail then runs as a decode split against a cache the
   chunk did not fill. Registering the chunk's experts (it already counts the routing) would turn the tail's
   ~10-15 GiB of reads into hits. Gate: tail chunk ≤ 0.25 s (from 0.81 s) and `doc` TTFT≥2 ≤ 800 ms.
2. **PLE gather on the GPU.** Keep the 28 GB table in host memory, map it, and gather the rows for a chunk
   with one kernel per layer (the payload is `rows x 160 x 2 B` — a few MiB per layer, not 600 MB). Gate:
   the per-row term ≤ 0.3 ms/row (from 0.46), i.e. `doc` turn-1 ≤ 3.6 s.
3. **Only read the experts a chunk needs.** Today the stream re-reads every routed expert per chunk; a chunk
   that follows a warm cache should read only what is not resident. This is item 1 generalised, and it is what
   makes `short`'s 1.19 s and `doc`'s 1.15 s approach strata's 0.63 s.
4. **The decode host term** (`dry = 1.0 + 0.25 ms x rows`, M16 item 4): the only lever left on TPOT
   (45 ms against strata's 39).

Order: **2** (the per-row tax) first now that 1 is ruled out, then 4 (the TPOT host term). Item 3 stays open
but needs a *design*: "read only the misses" is exactly what the small-chunk path already does, and the
measurements above say the win is not in the transport but in needing fewer bytes at all.

## PLE: no residency, memory map only (done, 2026-10-21)

Per the decision that PLE's per-row cost has little headroom and that it should simply be "memory, else disk",
the two ways of reaching the device were deleted (`mistralrs-core`, 295 lines removed, 20 added):

| removed | what it was |
|---|---|
| `PleTable::make_resident` + `ResidentTable` + `copy_iq4_nl` + `quantize_shards` | a device copy of the 28 GB table "if it fits", in a gather-kernel layout |
| `PleTable::device_index` + `device_reads_pageable_memory` | a pointer table so the GPU could gather rows straight out of the host mapping |
| `PleLayer::gathers_on_device` (now `false`) | the switch that chose between them; it also gated decode-graph capture, which a host gather can never replay |

What replaced them is nothing: `load` keeps the memory map (`TableSource::Gguf`), `prepare_table` only calls
`advise_random_access`, and `gather_host` reads the rows - out of the page cache when it holds them, off the
disk when it does not. Startup now says so once:

```
Qwen4-Exp PLE rows are gathered from the memory-mapped table on the host (page cache, else disk)
```

**Baseline with `PLE_DBG=1`** (the probe already existed; it sits on the host gather, so it covers prefill):

| measurement | value |
|---|---|
| per gather call | **5.7 ms** |
| rows per call | 638 (average over a mixed run) |
| **layers carrying a PLE** | **one** (`ple_cfg.layer_idx`), not one per decoder layer |
| share of `doc` turn-1 | one gather of 1 725 rows ~= **5.7 ms of 4.5 s (0.1%)** |
| decode | one call per step (one row), on the same host path |

**Correcting this document's own first guess**: PLE is ~0.1% of a turn, not the bulk of the per-row term. The
0.46 ms/row is therefore the QSA attention and the `hc_*` hyper-connections, and the PLE item is closed rather
than deferred - there is nothing to win there, which is also what the decision to leave it on the memory map
rests on.

Oracle re-verified after the deletion (byte-identical, both variants), 36/36 tests, `doc` turn-1 4.58 s
(unchanged, i.e. removing the residency attempt cost nothing - it had always fallen back).

## naiwa vs strata: architecture and performance (2026-10-21)

### Performance, 3 rounds, medians

| metric | naiwa | strata | ratio |
|---|---:|---:|---:|
| `doc` turn-1 TTFT | 4 431 ms | 2 430 ms | 1.8x |
| `doc` TTFT >= 2 | 1 153 ms | 631 ms | 1.8x |
| `long` turn-1 TTFT | 4 286 ms | 2 273 ms | 1.9x |
| `long` TTFT >= 2 | 1 062 ms | 1 018 ms | 1.04x |
| `short` turn-1 TTFT | 1 065 ms | 1 035 ms | 1.03x |
| `short` TTFT >= 2 | 1 185 ms | 625 ms | 1.9x |
| TPOT (all classes) | 45-46 ms | 39 ms | 1.15-1.18x |
| prefix reuse `doc`/`long` | 0.98 | 0.99 | parity |
| `e2e`/conv, `long` | 38.1 s | 39.9 s | **faster** |

So: reuse is at parity, `long` and `short` turn-1 are at parity, `long` end-to-end is faster, and what is left
is the first-turn prefill (1.8-1.9x on `doc`), the per-turn prefill (`doc`/`short` 1.8x), and TPOT (1.15x).

### Architecture, dimension by dimension

| dimension | naiwa | strata |
|---|---|---|
| language / stack | Rust on a `mistralrs` fork; candle CUDA kernels; our thread and stream discipline (doorbell rings, one copy stream per device, rayon expert pool) | C++/CUDA, hand-written kernels (MMQ, fused MoE, WMMA GEMM), captured graphs (`moe_grouped_gemm_*`, `s2_expert_grouped`, `mmq_resident_sort_rows`) |
| weights on the device | 4 300 MiB, at the file's own cost (verified by `MISTRALRS_VRAM_DUMP`: 2 838 quantized verbatim + 1 299 dense that are BF16 *in the file*) | pack weights 1 406 MiB loaded from its own pack + 708 MB of drafter experts |
| prefill chunking | step ceiling from `--max-num-batched-tokens`/`--max-prefill-chunk-tokens` (8 192), free-VRAM plan picks the largest chunk that fits, rows sub-batched at 256, plus a split at the checkpoint boundary | "auto chunk scan" over a byte budget (`set_ring_budget`), `stream_all_min()` = 1 024 selects MMQ vs the fused walk, last chunk is an ordinary remainder |
| prefill expert supply | prefill stream: whole-layer DMA into one or two 650 MiB scratch buffers on a copy stream, **cache-blind** | resident experts read from the VRAM tier, the rest streamed through a ring (8 slots below 1 024 tokens, `ring_slots()` above), **grouped by expert**; optional CPU share below 1 024 (-20..-35%) |
| decode expert supply | per-layer slot cache (LRU + hit counts) with lookahead prediction, 16-slot staging ring for `--pcie-frac`, CPU rayon pool for the rest | expert cache 1 028 slots / 1.98 GiB, hits on the device ("decided on the device"), 47.3% decode hit rate, 261 experts per token read over PCIe, CPU pool for the rest |
| prediction | one lookahead thread per model: the CPU router runs on the published activation and prefetches the next layer's experts | per-layer "foresight" prefetch of the next layer's slots (`foresight swap: N slots per layer`) |
| drafter (MTP) | built-in MTP head; its 512 experts **resident in Q2_0 (675 MiB)** since M16, GPU for both phases, host copy dropped | MTP layer with its **512 experts resident (708 MB, 2.25 bit)**, Q8_0 projections, dense attention over seen cells, grouped hit kernels, draft-vocab subset, optional `--mtp-q4` |
| prefix reuse | recurrent snapshot at paged **block boundaries**, captured during prefill and (since M16) during generation, alignment forced by committing the verify step to the boundary; restore is hash-keyed and block-aligned | **dense per-session KV** + `conversation_kv_save/restore(upto)`: any length, so the turn's exact end is the checkpoint and there is no alignment tail |
| PLE | 28 GB table memory-mapped from the GGUF, gathered on the host once per step (single PLE layer, 5.7 ms/chunk, 0.1% of turn-1) | the table lives on the **SSD** with an asynchronous per-token pipeline (`ple_issue_token` / `ple_finish_token`: issue the next token's rows while the current one is consumed, H2D after) |
| KV cache | paged, 32-token blocks, BF16, recurrent (GDN) state per sequence slot (114.8 MiB/slot, `--max-seqs` caps it) | per-session dense QSA pools + indexer, `--kv int8` available, elastic sizing |
| scheduling | mistralrs scheduler: `--max-seqs`, 8 decode steps before a prefill, `--prefix-cache-n` snapshots | own session/plan: `--max-context`, `--spec`, elastic KV growth, batched slots |
| specialization | one model (Qwen3.8-Flash-Next) end to end, so every path can assume it | same, but exposed as a general engine with several model families |

### What that says about the two remaining gaps

1. **First-turn prefill (1.8-1.9x).** Both engines read the same ~23 GiB (a chunk above ~26 rows spans every
   expert). We do it as whole-layer DMA at ~8 GB/s (cache-blind, one buffer on this card); strata does it as a
   grouped, resident-aware walk with a bigger ring, and its CPU share takes the tail. That is a *transport and
   selection* difference inside the same physical floor - our measured 2.93 s fixed term is 23 GiB at 8 GB/s,
   and 13.4 GB/s would make it 1.75 s.
2. **TPOT (1.15x).** Ours carries `dry = 1.0 + 0.25 ms x rows` of host time per layer (one round trip per layer,
   the row-proportional part being dispatch). strata decides residency on the device and replays a captured
   token graph, so it has no per-layer host round trip at all. This is M16 item 4 / M17 item 4, and it is the
   only lever left on TPOT.
3. **Why `doc`/`short` per-turn prefill is still 1.8x**: the tail. Our checkpoint needs a block boundary, so a
   turn ends with a <=31-row fragment that must read a layer's experts again; strata's dense KV needs no such
   fragment. Fixing that is a KV-layout decision, not a scheduling one.

## Gates

- Oracle unchanged: `frac 0 + --no-lookahead` byte-identical to `harness/m1/reference.oracle.txt`, with and
  without `--prefill-stream`; 36/36 tests (M16 baseline).
- One mechanism per measurement run; medians of ≥ 3 rounds; `TTFT`, `TPOT` and `reuse` are the comparable
  columns — `e2e` is not, when the generated lengths differ between runs (M16 saw a 400-token-cap round move
  `long`'s e2e by 15 s with no code change).
- Every item states the chunk-level measurement it must move, not just the class median.
