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

---

## 9. S1, first attempt: the truncation at `prompt_chunks.rs:274` is **not** the only cap

The first slice was the cheapest possible experiment: a flag (`NAIWA_NO_TAIL_SPLIT=1`) that makes
`build_prompt_chunk_plan` leave the **final** chunk untruncated, so the plan for a 3 143-token prompt becomes
`[0..3143]` in one chunk. Built, run, and the flag does what it says:

```
NO_TAIL_SPLIT pos=0    end=3143 total=3143     <- the plan, with the flag
NO_TAIL_SPLIT pos=3136 end=3143 total=3143     <- and a second chunk starting at 3136
```

Two things follow, and they redirect S1:

1. **The executed prompt step is still 3 136 rows** (the prefill stream logs `rows 3136`, and `NAIWA_STEP_TIME` still
   shows three `prompt=true` steps with the third at ~346 ms). So something other than the chunk plan caps the step -
   the plan asks for one 3 143-row step and the engine runs a 3 136-row one.
2. **A second chunk `[3136..3143]` exists in the plan**, i.e. the planner is being invoked more than once for this
   request (a scheduler-visible pass and then a re-plan), and one of those passes still trims at the block boundary.

The most likely second site is the **prompt token budget**, which is expressed in KV blocks
(`chunk_size = min(max_prefill_chunk_tokens, token_budget)`, and a budget derived from free blocks is inherently a
multiple of 32) - that would make the step's length block-aligned no matter what the chunk plan says. Finding it is the
first task of S1, and it is a grep plus a print, not a redesign.

**What stands from this attempt**: the flag is in the tree and documented (it is a no-op in practice today, which is
exactly what makes it a good bisection arm), and M35's centre of gravity moves from "the chunk plan truncates" to "the
prompt step's length is block-quantised in the scheduler's budget". The design (tail-aware key) is unaffected - it still
removes the *need* for the boundary; what changed is where the boundary is imposed.

## 10. The leading candidate now: it is KV blocks, not the checkpoint

Following §9's "a second cap exists", `prefill_token_budget` is
`min(max_num_batched_tokens, max_prefill_chunk_tokens, PROMPT_STEP_CEILING)`, none of which is 3 136 - yet two plan
passes appear, one ending at 3 143 (with §9's flag) and one ending at 3 136. And 3 136 = 98 x 32.

Which points at the one thing in this engine that is *inherently* 32-quantised: **the KV blocks a prompt step can
allocate**. If, with `--prefix-cache-n 1`, the block retention that belongs to the checkpoint keeps blocks alive, the
scheduler has **98** blocks to hand out instead of 99, the prompt step caps at 98 x 32 = 3 136, and the 7-token remainder
becomes its own step - with `-n 0` the same prompt runs as one 3 143-row step, which is what we measured all along.

That fits every observation in §25-§27 and needs no new mechanism:

| observation | explained by "98 free blocks" |
|---|---|
| `-n 0` runs 3 143 rows, `-n 1` runs 3 136 | one block fewer free |
| `--prefill-stream-min-tokens 1` changes nothing | it is not the stream threshold |
| MTP off changes nothing | it is not the verify path |
| the extra step costs 346 ms | a full 48-layer pass with the cache the loan emptied |
| no GPU/API difference in a profile | it is an ordinary step |

**If this is right, M35's premise is incomplete**: the tail exists because of *block capacity*, and the checkpoint's
alignment only coincides with it. Then the fixes are different - free one block for the step (do not retain the block
that the tail will fill), or let the step span "98 blocks + part of the next", which is exactly the partial-block
support M35 §3 item 3 asks for anyway.

The next probe is one line: print the scheduler's free-block count and the computed token budget at the prompt step for
both arms. Until that lands, the honest statement is that this milestone has located the cost (§27) and *one* of its
causes (§9), and that the second cause is now a specific, testable hypothesis rather than an unknown.

## 11. The probe: it is not KV blocks (they are 128 free), and the plan/step mismatch is now the question

`NAIWA_BUDGET_DBG=1` prints the scheduler's free-block count and the token budget at every prompt batch:

| arm | call | free blocks | ceiling | chunk_size | block alignment |
|---|---:|---:|---:|---:|---|
| `-n 1` | 1 | **128** | 0 | 8 192 | **true** |
| `-n 1` | 2 | **128** | 3 840 | 3 840 | **true** |
| `-n 1` | 3 | 30 | 1 792 | 1 792 | **true** |
| `-n 0` | 1 | **128** | 0 | 8 192 | **false** |
| `-n 0` | 2 | **128** | 3 840 | 3 840 | **false** |

Two conclusions, one of them a falsification:

1. **§10's "one block fewer free" is wrong.** Free blocks are **128** in both arms at the moment the prompt is planned;
   retention does not cost a block. What differs between the arms is exactly one thing: `block alignment` true against
   false. So the tail is the *alignment*, not the capacity - which restores M35's premise from §3.
2. **The plan and the executed step disagree, and that is now the open question.** With the 3 840 budget, `end` is
   `min(3840, 3143) = 3143`, and §9's flag makes the planner *keep* 3 143 (the marker printed `pos=0 end=3143`), yet the
   step that runs has **3 136** rows and a second plan `pos=3136 end=3143` appears. So the row count of the executed
   prompt step is decided somewhere other than this plan - the candidates, in order: `materialize_prompt_chunks`
   (which allocates KV per chunk from `chunk.end`), `seq.set_prefill_toks(..chunk.end)` /
   `computed_updates`, or the scheduler-visible-chunk pass that re-plans from `num_computed_tokens`.

Note 3 136 = `align_down(3143, 32)` exactly, which is why it kept looking like the alignment in the plan: with
`chunk_size = 3 840` the plan's own truncation *and* whatever the step does both land on 3 136, and a flag that disables
only the first is invisible in the result. **To separate them the probe has to print the executed step's row count and
its source**, not the plan's.

So the state of this milestone: the cost is located (§27), the mechanism is `block alignment` and not capacity (§11,
falsifying §10), and the remaining unknown is narrow - *which* of the three places above decides the step's 3 136 rows.
Everything else is designed and its feasibility checked (§3-§4).

## 12. S1 measured: the tail is gone and 483 ms with it - the real site was the *reusable-boundary cap*

Two truncation sites existed in the same function, and §9's flag only disabled the first:

| line | what it does | disabled by §9's first attempt? |
|---|---|---|
| `prompt_chunks.rs:280` (`aligned`) | clamps `end` down to the block boundary | yes |
| **`prompt_chunks.rs:163/169` (`reusable_boundary` + `cap_at_reusable_boundary`)** | **caps `end` at `align_down(total_len, 32)` so the checkpoint boundary is a chunk end** | **no - and this is the one that fires** |

So the flag now disables both for the **final** chunk (`cap_boundary` in `build_prompt_chunk_plan`), while non-final chunks
keep both (their ends are chunk boundaries anyway, which is where the reusable checkpoint lives). Same prompt, same
session, two runs an arm, `NAIWA_STEP_TIME=1`:

| arm | TTFT | the MoE step sequence |
|---|---:|---|
| baseline | **3 751 / 4 049 ms** | `seq 1, 3136, 7, 1, 5, 5` |
| `NAIWA_NO_TAIL_SPLIT=1` | **3 490 / 3 343 ms** | `seq 1, **3143**, 1, 5, 5` |

**The 7-row step is gone, and the TTFT median falls 3 900 -> 3 417 ms (-483 ms, -12%).** The step count is the
structural evidence: three `prompt=true` steps become two, and the prompt step carries all 3 143 rows.

Two honest notes:

1. **A harness artefact nearly hid this.** The first run of the "baseline" arm was accidentally *also* flagged, because
   `env NAME=` in the measurement script **defines** the variable and `var_os` is a presence test - the same trap
   `Slab::borrowing` documents ("a presence test made `=0` mean on"). Both arms printed `rows 3143`, which is what
   exposed it. The script now only exports variables that are non-empty.
2. **This flag alone trades the checkpoint away.** With the tail gone the turn has no block-aligned end, so no checkpoint
   is stored and the next turn cannot reuse - it is M34 §28's "fix 2", worth 483 ms for a single-shot request and a loss
   for a conversation. That is exactly what M35's tail-aware key (§3) exists to fix: keep the checkpoint, keyed on the
   prompt's real end.

Gates on the shipped defaults (flag off): `cargo test` 56 passed, `frac 0` + `--no-lookahead` byte-identical
(the oracle arm runs `--prefix-cache-n 0`, where `block_align` is `None` and neither site is reachable).
