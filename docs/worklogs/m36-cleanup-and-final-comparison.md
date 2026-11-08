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
