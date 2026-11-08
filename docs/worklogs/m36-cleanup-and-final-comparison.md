# M36 — the tree after the cleanup, and the comparison that says where the gap is

**Status:** ✅ done (2026-11-07). **Parent:** M35.

M35 ended with a fix that worked (the prompt's last chunk keeps its natural end) wrapped in a mechanism that did not (a
tail-keyed prefix checkpoint, captured but never restored). This milestone deletes the mechanism, keeps the fix, and
re-measures everything that matters, so the repo's numbers describe the code that is actually in it.

## 1. Why the tail checkpoint was deleted, in one table

Three arms on the same 3 143-token prompt, one request each, `NAIWA_STEP_TIME` on. `NAIWA_NO_TAIL_SPLIT` is "keep the
last chunk's real end, store nothing"; `NAIWA_TAIL_CHECKPOINT` is "keep the real end *and* store a tail-keyed
checkpoint"; the default is the old behaviour:

| arm | prompt steps | TTFT | whole request |
|---|---|---|---:|---:|
| default (truncates to 3 136, then a 7-token step) | 437, 2 999, **335 / 323** | 3 485 / 3 465 ms | 4 149 / 3 904 ms |
| `NAIWA_NO_TAIL_SPLIT=1` | 438, 3 036 | 3 182 / 3 177 ms | 3 766 / 3 604 ms |
| `NAIWA_TAIL_CHECKPOINT=1` | 441, 3 036 | 3 045 / 3 186 ms | 3 617 / 3 643 ms |

The two fixed arms are the same measurement. The tail checkpoint bought **nothing** — it could not, because the restore
side that would consume it is default-off for a reason found in M35 §32: enabling it produces **no tokens at all**, and
the reuse the chat harness actually gets comes from the *aligned* checkpoints `MISTRALRS_ALIGN_CHECKPOINTS` takes during
decode (`cached` stays 1792…2240).

So the whole capture/restore apparatus — `tail_hash`, `partial_block_key`, `register_partial_block_hash`,
`register_partial_block_hash`'s retention path in `block_pool`, `PagedTail`, `add_paged_recurrent_prefix_tailed`,
`peek_longest_paged_recurrent_prefix_tailed`, the scheduler's tail extension, the engine's tailed gate and its
diagnostics, plus the `NAIWA_TAIL_CHECKPOINT` / `NAIWA_NO_TAIL_SPLIT` / `NAIWA_TAIL_RESTORE` knobs — was deleted.

## 2. What the fork is now

`m35-cleanup`, branched from the commit that first stopped the truncation, with two later commits picked on top:

| commit | what |
|---|---|
| `0021227e` | the prompt's final chunk keeps its natural end (both truncation sites) |
| `500a6391` | the tile mma attention kernel is the default (oracle stays byte-identical) |
| `e0a747d5` | `div_ceil` for the computed block count |
| `16cedeb7` | the last chunk's alignment folds into the same rule |
| `8602205a` | deletes the tail machinery, makes no-truncation unconditional, drops `CHUNK_DBG`/`BUDGET_DBG` |

The unsafe restore path is not on the shelf any more; it is gone. `backup-m35-full` / `m35-full` keep the old head if it
is ever wanted.

On the naiwa side the same pass removed `NAIWA_REFILL_NOSYNC` (the §35 probe that ruled the refill wait out) and the
`NAIWA_TAIL_CHECKPOINT` env entry from `harness/chat/engines/naiwa.json`.

## 3. The comparison, measured after the cleanup

Same session, arms interleaved. Wall clock on this host carries ±100 ms of drift between runs (host memory pressure), so
each metric lists both runs; kernel time is the host-independent one.

### 3.1 A cold 3 143-token prompt

| metric | naiwa | strata | gap |
|---|---:|---:|---:|
| TTFT | 3 248 / 3 378 ms (median 3 313) | 3 209 / 3 244 ms (median 3 227) | +86 ms / **+2.7%** |
| whole request (8 tokens) | 3 840 / 3 896 ms | 3 517 / 3 561 ms | +329 ms / +9.3% |
| prefill step | ~3 040 ms | 3 145 ms / 999 tok/s (its own log) | **parity** |
| **kernel time** | **1.86 s** | 1.97 s (historical, §21) | **−5.6%, we are faster** |

The kernel total improved from 1.98 s to **1.86 s** with this cleanup alone: the deleted path hashed the tail, retained a
block and snapshotted recurrent state at an unaligned boundary on every prompt, and none of it was ever read.

Kernel breakdown of the 1.86 s (`nsys`, 3 143-token prompt):

| component | time | share | launches |
|---|---:|---:|---:|
| MoE/dense quant GEMM | 0.819 s | 44.0% | 2 122 |
| linear-attn (GDN) | 0.279 s | 15.0% | 701 |
| attention (tile mma) | 0.160 s | 8.6% | 245 |
| other | 0.153 s | 8.2% | 11 924 |
| CPU-expert wait (doorbell) | 0.116 s | 6.2% | 470 |
| bf16 GEMM | 0.107 s | 5.8% | 1 571 |
| hyper-connection | 0.089 s | 4.8% | 1 455 |
| MoE route/reduce | 0.084 s | 4.5% | 1 711 |
| cast/copy | 0.055 s | 3.0% | 3 241 |
| PLE/embedding gather | 0.001 s | 0.0% | 51 |

### 3.2 Decode on its own

23-token prompt, 128 generated tokens, so prefill is ~0.6 s of a 6 s request:

| arm | 128 tokens | per token |
|---|---:|---:|
| naiwa | 6 434 / 6 178 ms | **46.0 / 43.5 ms** |
| strata | 6 327 / 5 718 ms | **43.3 / 39.5 ms** |

Median 44.8 ms against 41.4, **+8.2%** — and that is with our MTP drafting while strata drafts nothing
(`drafts accepted 0 of 0` in its log). It is also the *only* place a gap remains.

## 4. Where the gap is, in one line each

| | state |
|---|---|
| prefill | parity — 999-1 003 tok/s both, and our kernel total is now 5.6% *lower* |
| TTFT, cold long prompt | +2.7% |
| decode | +8.2% per token (and up to +35% over the first few steps, M35 §37) |
| whole request | +9.3%, i.e. entirely the decode half |

The next step is unchanged from M35 §37 and now has no M35 machinery in the way: profile a **decode-only** request in
both engines and diff the kernel mix at one row per step. Neither side has ever been profiled that way.

## 5. Gates

| gate | result |
|---|---|
| `frac 0` + `--no-lookahead` oracle | **byte-identical** to `harness/m1/reference.oracle.txt` |
| `cargo test` (naiwa) | **56 passed**, 0 failed |
| chat harness, 4 corpora, 7 rounds | **154 turns, 0 empty outputs**, reuse median 0.94 |
| build | naiwa release, fork `mistralrs-core --features cuda` — both clean |

Note for the oracle: `capture_oracle.sh` now also prints `captured -> …` on its second line, so compare the text line
(`head -1`), not the whole file. The invocation that reproduces the reference is
`MODEL=~/Models/qwen3-coder/IQ1_M BIN=$PWD/target/release/naiwa-rs FRAC=0 NO_LOOKAHEAD=1`.

## 6. Where the decode gap comes from: the GPU is waiting on the CPU, and it is half the time

A decode-only request profiled with `nsys` (23-token prompt, 128 generated tokens, shipped config). Kernel total
**3.44 s** over 128 tokens = 26.9 ms of kernel per token, 183 412 launches = **1 433 launches per token**.

| kernel | total | share | launches | per launch |
|---|---:|---:|---:|---:|
| **`doorbell_wait_copy_f32_kernel`** | **1.692 s** | **49.1%** | 2 627 | **644 µs** |
| `mmvq_gguf_q6_k_bf16_plain_cuda1` | 0.178 s | 5.2% | 357 | 498 µs |
| `gemv_kernel_batched<bf16>` | 0.148 s | 4.3% | 15 909 | 9.3 µs |
| `moe_gemv_down_aggregate_iq4nl_q8_1` | 0.146 s | 4.3% | 4 485 | 32.7 µs |
| cutlass wmma bf16 | 0.134 s | 3.9% | 8 829 | 15.2 µs |
| `moe_gemv_fused_gate_up_iq2s_q8_1` | 0.117 s | 3.4% | 2 292 | 50.9 µs |
| `mmvq_gguf_q6_k_bf16_plain_cuda3` | 0.115 s | 3.3% | 2 856 | 40.1 µs |
| `moe_gemv_fused_gate_up_iq3xxs` | 0.104 s | 3.0% | 1 959 | 53.1 µs |
| `doorbell_copy_f32_kernel` | 0.048 s | 1.4% | 2 679 | 17.8 µs |
| `gdn_speculative_recurrence_*` | 0.042 s | 1.2% | 1 980 | 21.5 µs |
| `q4_qsa_attention_kernel` | 0.034 s | 1.0% | 824 | 41.7 µs |

The whole MoE GEMM family — the actual expert arithmetic, `moe_gemv_fused_gate_up_*` + `moe_gemv_down_aggregate_*` —
sums to **0.75 s, 5.9 ms/token**. The GPU spends **13.2 ms/token doing nothing** while it waits for the CPU experts: that
is 30% of the 45 ms wall clock, and it is 30x the cost of all attention kernels in the same window.

2 627 waits over ~51 steps of 48 layers is one wait per layer: at a 2.5 GiB cache almost **every layer misses** at least
one of its ten experts, so the CPU path runs on essentially every layer.

### The perfect-cache ceiling says the compute path is not the problem

`NAIWA_GPU_ONLY_EXPERTS=1` drops every non-resident expert once the cache is full - lossy, output is garbage, but it
bounds what decode would cost with a cache that never misses:

| arm | per token |
|---|---:|
| shipped (CPU experts for the misses) | 45.1 ms |
| **`NAIWA_GPU_ONLY_EXPERTS=1` (perfect cache, lossy)** | **26.4 ms** |
| strata | 39-41 ms |

So the CPU round trip costs **~19 ms/token**, and with it removed decode would be **faster than strata by a third**. The
deficit is not in our GEMMs, our attention or our MTP — it is that we do not hide the host's expert latency, while strata
does (its log: `CPU pool tasks/phase: 18`, `participating threads: 6`, and a 33.8% hit rate on a *smaller* 1.98 GiB
cache).

### 6.1 What that means for the next milestone

Three levers, in order of measured size:

1. **Overlap the CPU batch with GPU work** (~13 ms/token available). Today the token thread submits a layer's misses,
   then joins them at `BudgetPhase::CpuJoin` before that layer's MoE can run, so the GPU idles inside
   `doorbell_wait_copy_f32`. strata's equivalent batching runs 18 tasks over 6 threads and keeps the host ahead.
2. **Raise the hit rate** (the same 13 ms is what a bigger or better-profiled cache buys): 2.5 GiB of 23.4 GiB today.
3. **Cut launches** - 1 433 per token, of which 15 909 `gemv_kernel_batched` (124/token) and 12 234 hyper-connection
   launches are candidates for fusion, and 20 783 cast/copy for elimination.

Note the trap this rules out: it is tempting to read the doorbell's 644 µs as "the CPU is slow". It is not - the CPU
computes a layer's seven-ish experts in that time on AVX-2 with a 10-thread rayon pool, which is the same work and the
same ISA strata uses. The difference is *when* it is asked: strata asks earlier and in batches, we ask and then spin.

## 7. Correcting §6: the hit rate is 80%, and deeper MTP does not amortise the wait

Two measurements that sharpen §6, taken while explaining it.

**The cache is not cold.** `NAIWA_STATS=1` on the same 128-token decode: **80.3% hit rate** (68 322 hits, 16 726 misses,
18 632 staged, 2 277 promotions), lookahead `470 loaded / 15 640 swaps`. So a layer receives ~10 lookups and misses ~2-3
of them; with one lookup in ten missing, 1 - 0.8^10 = 89% of layers take the host path - which is what "2 627 waits over
~51 steps of 48 layers" measures. §6's "almost every layer misses" holds; "the cache contributes almost nothing" does not.

**Deeper speculation makes it worse, not better.** The hope was that a verify step's rows share the same 48-layer chain,
so the host batch would amortise over more tokens:

| `--mtp-n-predict` | per token |
|---|---:|
| 1 | 44.1 ms |
| 2 (default arm) | 42.3-45.1 ms |
| 3 | 49.8 ms |

It does not amortise, because the miss set per layer is the **union over the rows**: three draft rows route to three sets
of experts, so the host batch for one layer grows with the row count while the chain length does not shrink. Amortisation
would need the same experts across rows, which is precisely what drafting destroys.

### What 644 µs actually is

The work behind one wait is ~2.7 missed experts x 3 projections of 640x2560 = 12.4 M MAC = ~25 MFLOP, which at a
realistic 40 GFLOP/s for AVX-2 quantised GEMV over 10 threads is **0.5-0.6 ms** - the wait *is* mostly the compute, not
handshake overhead. So the honest statement is not "the CPU is slow" but "the CPU is busy doing real work while the GPU
has nothing else to do", which is a **scheduling** problem: the same work is done on the critical path, once per layer,
48 times per token.

strata does *more* host work than we do - its long-prompt decode reports a **33.8%** hit rate against our 80.3% - and is
still 12% faster per token. That is the whole argument: the work is unavoidable, the exposure is not.

### Why "submit, do unrelated GPU work, join late" is not available inside a step

The chain inside a decode step is strict: `attn_L -> route_L -> MoE_L -> x_{L+1}`. Layer L's routing is what names the
experts, and layer L's output is what layer L+1 needs, so between the submit and the join of one layer's misses there is
**no GPU work that does not depend on them**. M37's original wording implied otherwise; it was wrong.

The only axes that expose the host work to overlap are therefore:

1. **Cross-step prediction** - compute step n+1's *predicted* misses while the GPU finishes step n. The machinery exists
   (the lookahead thread) and is empirically idle: 470 loads against 15 640 swaps. This is the axis to fix, and its
   quality metric is prediction accuracy, not draft acceptance.
2. **A smaller tail** - a bigger or better-ranked cache so fewer layers miss at all (2.5 GiB of 23.4 GiB today; the
   profile provides the ranking, but the drop rate at 2 500 of 12 288 experts is what leaves 2-3 misses per layer).
3. **Fewer, larger waits** - batch the misses of several layers per handoff where the dependency allows (a layer's
   *attention* path does not need the previous layer's MoE when the model allows overlap of the residual stream, which
   this architecture does not).

## 8. The host path itself is slow: 228 µs per expert, and that is 40 ms/token

The user's objection to §7 is correct and it re-orders the whole plan: you cannot precompute the next layer, because
layer L+1's experts are named by layer L's own attention output. So any "overlap" scheme is a guess. That sent me back to
the numbers, and the CPU path turns out to be the thing that is actually wrong.

`FUSED_DBG` (already in the tree, one line per 480 CPU batches) on a 24-token decode:

```
FUSED_DBG calls rows/call pairs/call experts/call us/call us/pair us/expert gap_us/call
FUSED_DBG 480   5.94       11.14       8.85         2016.9   181.1    227.8     1399.6
```

| reading | value |
|---|---:|
| CPU time per layer (one batch of ~9 experts) | **2 017 µs** |
| per (expert, row) pair | 181 µs |
| **per expert** (gate+up+down) | **228 µs** |
| GPU time between two CPU batches (the gap) | 1 400 µs |

Two conclusions, both measured:

1. **The host is the bottleneck, not the GPU.** A batch costs 2.02 ms and the GPU only has 1.40 ms of work before it needs
   the result - the 0.6 ms difference is exactly the 644 µs average
   `doorbell_wait_copy_f32` spin from §6. 480 batches x 2.02 ms = **0.97 s per 24 tokens = 40 ms/token of host compute**,
   against a 43.9 ms/token wall clock: decode is a host-bound workload with a GPU attached.
2. **228 µs per expert is 3-5x what this CPU should need.** An expert is 3 projections of 640x2560 = 4.9 M MAC ≈ 9.8
   MFLOP, and this host's AVX-2 path measures ~21 GFLOP/s effective - a good AVX-2 quantised GEMV on 6 cores is 60-100+.
   The gap is per-weight dequantisation: these experts are IQ2_S/IQ3_XXS/IQ4NL/Q2_0, where every weight is a codebook
   lookup, and strata's own log says what it does about that: *"the expert kernels run on AVX-2 (`multi-token` for the
   i-quant gate/up rows)"* - it processes several rows at once to amortise the lookups. We do have rows to amortise
   (5.94 rows/call) and evidently do not use them that way.

### The levers that are now ruled out, with their numbers

| lever | measurement | verdict |
|---|---|---|
| deeper speculation (more rows to amortise) | depth 1 44.1, depth 2 42-45, depth 3 49.8 ms/token | worse |
| move misses to PCIe and compute on the GPU (`--pcie-frac`) | 0.4: 44, 0.7: 51.9, 1.0: 54.1 ms/token | worse - 131 experts/token is 262 MB/token against a 13.4 GB/s link |
| more promotions (policy) | 2 277 -> 6 461 promotions, hit rate 80.3% -> 80.5% | no effect; it is capacity, not policy |
| bigger cache (capacity) | 2.5 -> 3.5 GiB: hit 80.3% -> 84.3%, 44 -> 41.2 ms/token | real but shallow, and VRAM-bound |
| cross-step prediction | lookahead does 470 loads against 15 640 swaps | not a plan: it is a guess about the next token's routing |

### So M37 becomes: make the host expert kernel 2-3x faster

Target: **228 -> under 100 µs per expert**, i.e. 2.02 -> under 0.9 ms per layer. That puts the host below the GPU's 1.40 ms
per layer, at which point the doorbell spin disappears on its own (the GPU becomes the bottleneck again) and decode moves
from 44 towards the 26-30 ms/token the compute path already supports.

The work is the same thing strata did and documented: specialise the i-quant gate/up/down rows for multi-row batches, so
one codebook lookup serves every row that shares the expert, instead of re-deriving the weights per row. The pieces to
look at are `mistralrs-quant/src/gguf/fused_cpu.rs` (the rayon pool and its per-pair loop) and the per-type row kernels it
calls. No change to numerics is required or wanted: the arithmetic order stays as it is, only the loop structure moves -
which keeps the §"arithmetic order" constraint intact.

## 9. How fast the host path can realistically get: 1.4x, not 2-3x

§8 claimed a 2-3x target. That is too much, and the arithmetic says so: what is needed is not a fast host but a host that
is *faster than the GPU per layer*.

| | per layer | per token |
|---|---:|---:|
| host batch (today) | 1 922-2 017 µs | ~40 ms |
| GPU work available between batches (`gap`) | 1 272-1 405 µs | 26.4 ms (the perfect-cache ceiling) |

The GPU already runs at 26.4 ms/token; the host needs 40 ms. Making the host match the GPU is a **1.37x** improvement, and
that alone would put decode at ~30-34 ms/token - ahead of strata's 39-41. Anything beyond that buys little, because the
GPU becomes the limit. So the honest target is **1.3-1.4x**, not 2-3x.

### What the existing code already does, and what it is worth

The fork is not starting from zero: `mistralrs-quant/src/gguf/fused_cpu.rs` already has multi-row kernels for **Q2_0** and
**IQ4NL** (`NAIWA_NO_Q2ROWS`, `NAIWA_NO_I4NLROWS` switch them off). Disabling them on a 24-token decode:

| arm | µs/call | µs/pair | µs/expert | gap | per token |
|---|---:|---:|---:|---:|---:|
| control | 1 922 | 171.4 | 213.9 | 1 405 | 46.7 ms |
| without Q2_0/IQ4NL multi-row | 2 120 | 204.8 | 258.7 | 1 272 | 54.1 ms |

So the technique that §8 proposed as the milestone is **already applied to two of the six quant types and is worth ~19%**
of host time (7.4 ms/token). Extending it to the i-quant gate/up rows (IQ2_S, IQ3_XXS, IQ3_S - which have no multi-row
path) should be worth the same order: **10-20%, not 2-3x**. That is still most of what the 1.37x target asks for, but it
must be stated as 1.2-1.4x.

### And a caution about the measurement itself

The same `--mtp-n-predict` arms measured 49.8 / 46.4 / 35.5 ms/token on three separate 24-token runs - a ±10% spread that
is as large as some of the effects §7 and §8 quote. Decode arms from here on should be 128-token runs, several per arm,
with medians; the FUSED_DBG columns (µs/pair, µs/expert) are far less noisy than the wall clock and should be the primary
metric, since they measure the thing being changed.

### Revised M37

1. Instrument `fused_cpu` per quant type (µs/expert by type) - measure before writing code.
2. Extend the existing Q2_0/IQ4NL row-batching to the i-quant gate/up rows.
3. Target: host ≤ 1.4 ms/layer (1.37x from today), decode 44 -> 34 ms/token, arithmetic untouched.

## 10. M37 step 1 done: the host cost by quant type, and no villain

`NAIWA_FUSED_PHASE_DBG=1` (new, fork `fused_cpu.rs`) prints one line per CPU batch with the two passes split. Aggregated
over the 2 473 batches of a 128-token decode:

| gate/up | down | calls | experts | gate/up total | down total | µs per expert (gate/up) | µs per expert (down) |
|---|---|---:|---:|---:|---:|---:|---:|
| IQ3_XXS | IQ4NL | 738 | 5 830 | 0.771 s | 0.437 s | 132 | 75 |
| IQ2_S | IQ4NL | 769 | 4 555 | 0.564 s | 0.353 s | 124 | 78 |
| IQ3_S | IQ4NL | 451 | 2 479 | 0.370 s | 0.196 s | 149 | 79 |
| IQ2_S | Q2_0 | 252 | 1 383 | 0.172 s | 0.077 s | 125 | 55 |
| IQ3_XXS | Q2_0 | 155 | 851 | 0.115 s | 0.046 s | 135 | 55 |
| IQ4XS | IQ4NL | 54 | 416 | 0.047 s | 0.030 s | 113 | 72 |
| IQ3_S | Q2_0 | 54 | 395 | 0.057 s | 0.020 s | 145 | 50 |

**gate/up is 64% of the host's expert time, down is 36%**, and no type is pathological: every gate/up stack lands in
113-149 µs per expert, every down stack in 50-79. The effective throughput is **~45-50 GFLOP/s** (gate/up: 6.55 MFLOP in
132 µs; down: 3.28 MFLOP in 72 µs) - not great for six AVX-2 cores on a memory-bound GEMV, but nowhere near the 3-5x
deficit §8 implied. §9's 1.37x is the right order, and even that is at the optimistic end.

### Why row-batching cannot give more than ~20%

The same batches hold **5.86 rows but 8.99 distinct experts**: each expert serves **1.25 rows** on average. Batching rows
can therefore only save the weight traffic of the *shared* fraction, i.e. ~20% - which is exactly what the existing
Q2_0/IQ4NL row kernels are worth (§9: disabling them costs +19%). Extending them to the i-quant gate/up rows is worth
about the same, not more.

### `--pcie-frac` is already at its optimum, and for a reason worth knowing

| `--pcie-frac` | per token | hit rate |
|---|---:|---:|
| 0.0 | 57.0 ms | **59.5%** (41 847 misses, 0 staged, 0 promotions) |
| 0.2 | 45.9 ms | 70.9% |
| **0.4 (shipped)** | **44 ms** | 80.3% |
| 0.7 | 51.9 ms | - |
| 1.0 | 54.1 ms | - |

At 0.0 the hit rate *collapses*, because the staging path is also what feeds promotions into the cache (`staged ->
promoted`). So the DMA is not paying for itself with link time, it is paying for itself with residency. That closes the
last "just turn the knob" idea: 0.4 is the measured optimum of the two effects.

### Where this leaves M37

The measured levers, in order of size, are now all small and all known:

| lever | expected | status |
|---|---|---|
| i-quant gate/up row batching (extend Q2_0/IQ4NL's technique) | ~1.1-1.2x on 64% of host time | not built; ~20% ceiling by the 1.25 rows/expert measurement |
| thread-count / SIMD balance in `fused_cpu` | unknown, bounded by 45-50 GFLOP/s today | needs its own measurement |
| cache capacity | 3-4% hit per GiB, 2.5 -> 3.5 GiB bought 44 -> 41.2 ms | VRAM-bound |
| everything else | - | ruled out with numbers in §7-§9 |

That is the honest position: decode's remaining gap is a host-side efficiency problem of order 1.2-1.4x, not a structural
one, and the largest single piece (gate/up) has a hard 20% ceiling from row sharing.

## 11. M37 step 2: the host path taken to its floor, and what that floor is

Three optimisations were implemented and measured; only the instrumentation survives, because the other two do not pay.

### 11.1 Expert grouping in `gate_up`: neutral

Pairs are sorted by expert, so one task can own a whole expert's run of pairs and keep the weight row in L1 instead of
pulling it once per pair. Implemented (`NAIWA_NO_EXPERT_GROUPING=1` as the A/B arm):

| arm | gate/up | down | per token |
|---|---:|---:|---:|
| per-pair scheduling (before) | 129 µs/expert | 72 | 44.1 ms |
| one task per expert's run (after) | 130 µs/expert | 71 | 42.8 ms |

**Neutral.** The weight read is not the cost - these kernels are bound by the codebook unpacking (ALU), not by DRAM. So
the change was reverted rather than shipped as dead complexity.

### 11.2 Thread count: the default is already the optimum

| `NAIWA_CPU_MOE_THREADS` | gate/up | per token |
|---|---:|---:|
| 6 | 145 µs/expert | 46.5 ms |
| **8 (the default, `available - 2` capped at 8)** | **130** | **42.8-44.1 ms** |
| 12 | 128 | 45.9 ms |

8 threads is what the fork already picks, and the per-expert time at 12 is no better while the wall clock is worse.

### 11.3 A bigger cache: better in isolation, not shippable

`--expert-cache-gb 3.5` measured **41.2 ms/token** on a decode-only request (hit rate 84.3% against 80.3%). In the chat
harness it **fails**: `naiwa round 2 short turn 6 failed: model_error` - 12 GB of VRAM cannot hold the cache, the KV pool
and the prefill scratch once the conversation grows. Reverted to 2.5 GiB, and the harness is green again (154 turns, 0
empty outputs, TPOT median 40.4 ms).

### 11.4 Why the remaining ~1.2x is not reachable here

The only real technique left is **dequantise once, dot many rows** for the i-quant gate/up stacks (what strata's
`multi-token` i-quant kernels do). Its ceiling follows from two numbers already measured:

- each expert serves **1.25 rows** in a decode batch (5.86 rows, 8.99 distinct experts), so at most ~20% of the pairs can
  reuse an unpacking;
- gate/up is 64% of the host's expert time, so 20% of that is **~13% of host time, ~3 ms/token** - under this host's
  ±10% noise, and less than the 2.5 -> 3.5 GiB cache gave before it ran out of VRAM.

So the measured floor for the host expert path on this machine is **~130 µs per expert for gate/up and ~71 for down, at
~50 GFLOP/s aggregate**, and decode's floor with the shipped configuration is **42-44 ms/token** (26.4 ms of GPU work
plus the exposed host work that cannot be hidden on a strict layer chain).

Getting below that needs a change of hardware or of strategy, not of code:

| would need | why |
|---|---|
| AVX-512 (this host has none; strata's own engine refuses to run its i-quant kernel for the same reason) | the kernels are codebook-bound, and AVX-512 VBMI does the lookups 2-4x faster |
| more VRAM than 12 GB | residency is the only lever that removes work rather than speeding it up: every GiB is worth 3-4% hit rate |
| the GPU computing the misses | measured worse here: 262 MB/token of misses against a 13.4 GB/s link (`--pcie-frac` 0.7/1.0 = 51.9/54.1 ms) |

### 11.5 What ships from M37

| change | state |
|---|---|
| `NAIWA_FUSED_PHASE_DBG` (fork `fused_cpu.rs`) | **kept** - the instrument that attributed 64/36 to gate/up/down and produced every number in §10-§11 |
| expert grouping | reverted (neutral) |
| `--expert-cache-gb 3.5` | reverted (does not fit) |
| thread count | unchanged (already optimal) |

## 12. M37 step 3: dequantise-once is real on the host (-9.2%) and neutral on the wall clock

Built the multi-row i-quant kernels after all: `vec_dot_q8_k_x2` for **IQ2_S**, **IQ3_XXS** and **IQ3_S**, which unpack
one weight row once and dot **two activations** against it (the grid lookups, sign masks and scale vectors are weight-only,
so the second row costs two loads and two `maddubs`). `gate_up` uses them whenever one expert serves two or more rows;
`NAIWA_NO_X2=1` is the A/B arm and `NAIWA_NO_EXPERT_GROUPING=1` restores per-pair tasks.

Each kernel is checked against the two single-row dots it replaces (`vec_dot_x2_matches_two_single_dots`, one per type):
identical to 1e-4 relative, and the AVX2 path agrees with the scalar fallback.

### Host time: -9.2% on the gate/up pass

Aggregated from `NAIWA_FUSED_PHASE_DBG`, two 128-token runs per arm:

| gate/up stack | x2 off | x2 on | change |
|---|---:|---:|---:|
| IQ3_XXS | 147 µs/expert | 125 | **-14.7%** |
| IQ3_S | 151 | 140 | **-6.9%** |
| IQ2_S | 125 | 121 | -3.3% |
| IQ4XS (no x2 kernel) | 118 | 123 | +4.4% (noise) |
| **gate/up total** | **4.464 s** | **4.053 s** | **-9.2%** |

-1.6 ms/token of host compute. That is the size §9/§11 predicted (~2-3 ms/token) and it is real, not noise: the per-type
per-expert columns carry a fraction of the wall clock's spread.

### Wall clock: unchanged, and that is the finding

| arm | 128-token decode, four runs | median |
|---|---|---:|
| x2 off | 46.9 / 45.7 / 45.5 / 44.6 ms per token | **45.6** |
| x2 on | 45.6 / 45.1 / 46.4 / 46.4 ms per token | **46.0** |

Neutral to slightly worse, i.e. inside the ±2 ms/token spread this host has. With the phases measured in the same runs the
reason is arithmetic: host compute is ~24.6 ms/token (gate/up 2.0 s + down 1.15 s over 128 tokens), the GPU path is
26.4 ms/token, and the wall is 45 - so **the host and the GPU overlap by roughly 6 ms and the GPU is now the larger of the
two terms**. Removing 1.6 ms from the smaller term cannot move the sum by more than that, and it does not: it disappears
into the overlap.

So the earlier "40 ms/token of host compute" in §8 double-counted (it took `us/call` from `FUSED_DBG`, which includes
the activation quantisation and slot bookkeeping the phase split shows separately). The corrected decomposition is:

| term | per token |
|---|---:|
| host expert compute | 24.6 ms |
| GPU work (perfect-cache measurement) | 26.4 ms |
| overlapped | ~6 ms |
| wall clock | 45 ms |

### What that closes

M37's premise was that the host is the critical path and can be made faster than the GPU per layer. Half right: the host
*is* on the critical path layer by layer (that is the 644 µs spin), but at the token level the GPU and host overlap enough
that a 9.2% host improvement is invisible. Both terms now need to shrink for the decode number to move, and §11 has
already measured what each would cost (AVX-512 or more VRAM).

The kernels stay, default on, because they are free: three unit tests, no numeric change to anything the oracle covers
(single-row decode never takes the two-row path), and a strictly lower host cost.

## 13. The overlap question: 18.6 ms/token is exposed, and the window is one layer wide

Asked whether the overlap between host and GPU has room. It does, on paper, and the size is easy to state from §12's
numbers:

| term | per token |
|---|---:|
| host expert compute | ~20-24.6 ms |
| GPU path (perfect-cache measurement) | 26.4 ms |
| wall clock | 45 ms |
| **hidden by overlap** | **~1.7-6 ms** |
| **exposed** | **18.6 ms** (0.39 ms per layer) |

If the host work were fully hidden the decode would cost ~26.4 ms/token, i.e. **18.6 ms/token is the prize**. The question
is whether the window can be widened.

### Why the window is one layer wide

Inside a step the chain is `attn_L -> route_L -> MoE_L -> x_{L+1}`. The misses of layer L can only be submitted after
`route_L`, and their result is needed by `MoE_L`, so the GPU work available to hide them is exactly *layer L's own cached
experts* - the `Hit` GEMVs, ~1.4 ms per layer, which is the 1 272-1 405 µs `gap` column in `FUSED_DBG`. The host batch is
~1.3-1.9 ms for the same layer. **That is why the two are neck and neck, and why a 9.2% host improvement (§12) disappears
into the overlap instead of showing up.**

Widening the window therefore needs one of:

1. **more GPU work between submit and join** for the same layer - not available: everything else in the layer depends on
   the MoE result;
2. **cross-step prefetch** of the next step's predicted misses - the lookahead thread, which is empirically idle (470
   loads against 15 640 swaps);
3. **fewer misses** so the host batch fits inside the window (cache capacity; VRAM-bound, §11).

### What was tried: staging waves and the DMA share

The staging ring is the one mechanism that turns host work into *overlappable* GPU work: a share of the misses is DMA'd
into the ring and computed on the GPU alongside the residents. `NAIWA_STAGING_WAVES` splits a step's staged experts into
waves, each with its own event, so the GPU starts on wave 1 while the later waves' DMA is still landing (events, not
slots - no extra VRAM).

| config | decode-only, 128 tokens |
|---|---:|
| `--pcie-frac 0.4`, waves 1 (default) | 44.3 / 45.5 / 44.2 ms per token |
| `--pcie-frac 0.4`, waves 4 | **42.8 / 44.2 / 42.3** |
| `--pcie-frac 0.7`, waves 4 | 49.6 |
| `--pcie-frac 1.0`, waves 4 | 59.0 |

So the overlap axis *is* real, and 4 waves buy 1.5 ms/token on an isolated decode. But the shipped workload is the chat
harness, and there the same change **loses**:

| arm | chat harness, 154 turns |
|---|---:|
| waves 1 (default) | **TPOT median 40.2 ms** |
| waves 4 | 41.6 ms |

The waves interleave with prefill chunks and with prefix reuse, and `staging_wait` grows (the effect M26 recorded). So the
default stays at one batch, `NAIWA_STAGING_WAVES` stays as the documented knob, and the win is recorded for the
decode-only case rather than shipped.

Raising `--pcie-frac` past 0.4 remains strictly worse: at 0.7 and 1.0 the link time of ~131 experts per token (262 MB
against 13.4 GB/s) exceeds what the window can hide, and the ring is 16 slots shared by all layers.

### Net answer

- The overlap space is **18.6 ms/token**, but it is not reachable by scheduling alone: the window is structurally one
  layer's cached-expert work (~1.4 ms) against a host batch of ~1.3-1.9 ms.
- The measured overlap knobs move 1.5 ms/token in the lab and lose it back in the chat harness.
- Getting the 18.6 ms needs the host batch to shrink *below* the window everywhere (fewer misses via VRAM, or a faster
  host via AVX-512) or a working cross-step predictor - the two things §11 already identified, in that order.
