# M35 — the prefix checkpoint is keyed on the prompt's real end, not on a block boundary

**Status:** proposed; feasibility checked against the code, nothing built (2026-11-06). This file is the design and the
plan; sections are added as the steps land, the way M33/M34 were.
**Parent:** M34 §27-§28 (the `--prefix-cache-n 1` penalty is a whole extra step for a 7-token tail: 346 ms of a 3.7 s
single-shot prompt), M10 ("a checkpoint needs a block-aligned prompt step" - the decision M35 replaces), M16 item 1
(`align_down(prompt, 32)`).
**One mechanism:** the prefix checkpoint's **key** gains the prompt's tail, so the checkpoint is taken at the **true end of
the prefill** instead of at the last 32-token boundary - the prompt step is never truncated, the tail step never exists,
and the recurrent state that gets stored is the one the next turn actually wants.

---

## 1. The problem, measured

`--prefix-cache-n 1` costs **275-346 ms** of the first token on a 3 143-token prompt, reproducibly (M34 §25-§27), and
the mechanism is fully located:

| | |
|---|---|
| the prompt step processes **3 136** tokens instead of 3 143 | `pipeline/prompt_chunks.rs:274` aligns the chunk end down to the 32-token block |
| the 7-token remainder runs as its **own prompt step** | `NAIWA_STEP_TIME` shows three `prompt=true` steps and the MoE runs `seq 7` |
| that step costs **346 ms** | a full 48-layer pass, and the loan has emptied the expert cache, so every expert it routes to is fetched per expert |
| the checkpoint it buys costs **1.2 ms** | `NAIWA_SNAP_TIME`: flush 131 us + 36 gathers 1.04 ms + aux 2 us |

So 346 ms are paid for a 1.2 ms artefact, because the artefact has to be *labelled* with a 32-token boundary.

## 2. Why the label had to be a block boundary (M10's reasoning, and why it no longer holds)

`compute_block_hashes` hashes **whole blocks only** (`block_hash.rs:238`: `num_full_blocks = tokens.len() / block_size`),
so a reusable prefix is a whole number of blocks; two guards then refuse anything else - `pipeline/mod.rs:1931`
(capture) and `engine/mod.rs:465` (restore: `"unaligned_boundary"`). A recurrent state can only be photographed at the
end of a forward. Therefore: **the forward had to end on a block boundary.**

M10 priced that correctly at the time: an extra chunk of 7 tokens on a **CPU prefill at 11-13 tok/s** is ~0.6 ms, and
it bought the whole reuse story (a `doc` turn went 141.9 s -> 5.6 s). The decision was right and its cost model has since
inverted: one chunk is now **a 48-layer GPU pass over 24 GiB of streamed experts**.

## 3. What M35 changes

The recurrent state does not care about blocks at all. What needs to change is the **addressability of the stored
prefix**, and it is three edits plus one check:

| # | edit | where | why it works |
|---|---|---|---|
| 1 | a **tail hash**: the entry's key becomes `(block_hashes[..n], tail_hash, tail_len)` where the tail hash covers the final `len % 32` tokens | `paged_attention/block_hash.rs`, `prefix_cacher.rs` | the retention machinery is **hash-keyed**, not block-indexed: `block_pool.rs:175 retain(&[BlockHash])` and `:790 get_common_cached_block(block_hash, ..)` already handle any hash |
| 2 | **relax the two alignment guards** so `cached_tokens` may end mid-block | `pipeline/mod.rs:1931`, `engine/mod.rs:465` | nothing below them requires a multiple; they are policy, not mechanism |
| 3 | **copy-on-reuse of the tail block**: on a hit, the partial block's valid rows are copied (7 rows x 2 kv heads x 256 x 2 B = 7 KiB) and the new tokens are appended to the copy | `paged_attention/scheduler.rs` (`materialize_prompt_chunks`) | the reused block must not be written to by the second request; a 7 KiB copy is free, and it is the standard copy-on-write rule for a partial page |
| 4 | **verify** (done by reading, not by running): retention stores `RetainedPrefixEntry { hashes, .. }` and the pool refcounts `hash_ref_counts[hash]`, so a tail hash is retained exactly like a block hash | `paged_attention/block_pool.rs:175-215` | - |

**What M35 does not touch**: no CUDA kernel, no step semantics, no extra snapshot, no change to the prefill stream, no
change to the loan. Compared with M34 §28's "checkpoint lane" (fix 1) it is strictly smaller, and compared with "don't
align" (fix 2) it does not give up the reuse.

## 4. The trade it makes, stated honestly

Today's key is `{multiples of 32}`; M35's is `{multiples of 32} U {the prompt's true end}` - **plus, in the common case,
minus the aligned one**, because the truncation that produced it is gone.

| case | today | M35 |
|---|---|---|
| next turn of the same conversation (shares the whole previous prompt) | reuse 3 136, recompute 7 | **reuse 3 143, recompute 0**, and no tail step |
| another request sharing the first 3 136 tokens but differing in the last 7 | reuse 3 136 | **no reuse at that entry** |
| single-shot request | pays 346 ms for a checkpoint nobody uses | **pays nothing** |

The second row is the cost. Two ways to keep it, in order of preference:

1. **the hybrid rule**: align only where a chunk boundary **already exists** (a multi-chunk prompt has one at the end of
   each non-final chunk, and it is free there); take the true-end checkpoint otherwise. Then both keys exist and neither
   costs a step.
2. store both checkpoints for a single-chunk prompt (the aligned one requires the truncation, so this is only worth it if
   measurement shows the "shared system prompt" case matters - it is a chat benchmark question, not a design one).

## 5. Plan

| # | step | what it settles | how it is checked |
|---|---|---|---|
| **S1** | tail hash + `cached_tokens` allowed to end mid-block, **without** the copy-on-reuse (single-sequence case only) | that the addressability change works end to end | chat `short`/`doc` harness: reuse ratio must be **>= today's**, TTFT for a 3 143-token prompt must drop by ~346 ms, and the tail step must disappear from `NAIWA_STEP_TIME` (2 `prompt=true` steps, not 3) |
| S2 | copy-on-reuse of the tail block | that a second request sharing the tail cannot corrupt the first | two identical requests in one process, then a third that diverges at token 3 137: outputs must match the no-cache arm |
| S3 | the hybrid rule (alignment only at existing chunk boundaries) | that the "shared prefix, different tail" case keeps its reuse | a two-request test sharing 3 136 of 3 143 tokens |
| S4 | the oracle arm's `--prefix-cache-n 0` stays the oracle; the shipped chat config runs S3's rule; measure `doc`/`short` end to end | the default | 3 rounds, 0 failures, and reuse >= today |

## 6. Gates

- `frac 0` + `--no-lookahead` byte-identical to `harness/m1/reference.oracle.txt` (that arm has no prefix cache, so it is
  untouched by construction - M35 must not change it);
- `cargo test` 56 passed;
- chat harness 3 rounds, 0 failures, **reuse ratio not below today's** (`short` 0.90-0.97 per turn, `doc` 0.95-0.99);
- the tail step is **absent**: `NAIWA_STEP_TIME` shows two `prompt=true` steps for a 3 143-token prompt, and TTFT drops by
  275-346 ms;
- a second sequence sharing a tail must be byte-identical to the no-cache arm (S2).

## 7. Risks

1. **Hash determinism of the tail.** The tail hash must be a pure function of the tail tokens and the parent block hash,
   or the same prompt will miss. Same shape as the block hash, one length down.
2. **Writes into a reused partial block.** The copy-on-reuse in S2 is not optional for correctness if two sequences can
   share the prefix; S1 tests single-sequence only for that reason, and must not be shipped without S2.
3. **The MTP draft rows.** M10 documented a hazard: a block written during a *speculative* step can hold draft rows in its
   tail. Capture is prompt-only (`is_prompt`), so the stored tail rows are committed target tokens - but the check must be
   repeated for the tail hash, which is new code.
4. **Metrics and the retention monitor.** `paged_recurrent_bytes`, `num_retained_prefix_blocks` and the revocation monitor
   count blocks; a tail hash is one more entry, and the counters must not double-count or leak.

## 8. What success looks like

A 3 143-token prompt on the shipped chat config: TTFT **3.85 s -> 3.51 s** (the 275-346 ms M34 §27 measured), two prefill
steps instead of three, the checkpoint still taken (1.2 ms), and `short`/`doc` reuse unchanged or better - with the
oscillation that started this, `--prefix-cache-n 1`, no longer tradeable against TTFT at all.
