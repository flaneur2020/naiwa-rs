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
