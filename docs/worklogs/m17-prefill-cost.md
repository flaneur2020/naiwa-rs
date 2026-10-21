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

Order: 1 (cheapest, and it is the same cache mechanism M16 already uses), then 2, then 4. Item 3 is 1 done
properly and should be designed with it.

## Gates

- Oracle unchanged: `frac 0 + --no-lookahead` byte-identical to `harness/m1/reference.oracle.txt`, with and
  without `--prefill-stream`; 36/36 tests (M16 baseline).
- One mechanism per measurement run; medians of ≥ 3 rounds; `TTFT`, `TPOT` and `reuse` are the comparable
  columns — `e2e` is not, when the generated lengths differ between runs (M16 saw a 400-token-cap round move
  `long`'s e2e by 15 s with no code change).
- Every item states the chunk-level measurement it must move, not just the class median.
