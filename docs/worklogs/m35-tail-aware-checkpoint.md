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

## 13. What is settled, and the one comparison this ad-hoc harness cannot make

Settled and measured:

| | |
|---|---|
| the truncation site | **`cap_at_reusable_boundary` + `reusable_boundary`** (`prompt_chunks.rs:163/169`), not the `aligned` clamp at `:280` - §9's first attempt disabled the wrong one |
| disabling both for the final chunk | the `seq 7` tail step disappears and the layer-0 sequence becomes `1, 3143, 1, 5, 5` |
| TTFT effect | **3 900 -> 3 417 ms** median (3 751/4 049 against 3 490/3 343), **-483 ms (-12%)**, two runs an arm |
| the tail path's correctness | not the bug it looked like: the same prompt produces the same first text with and without the split (`'The text provided is a sequence of Greek letter names. To'` against `'... Greek letters. When decoded'`, i.e. a difference in wording only), and the 1 657-token prompt that produced **no** token does so in **both** arms - it is the prompt (a random-word salad), not the split |
| the KV-block hypothesis | falsified (§11): 128 free blocks in both arms |

Not settled: **the chat trade**. The flag alone drops the checkpoint, so the next turn should lose its reuse - but the two
ad-hoc two-turn runs could not show it, because both arms re-prefilled turn 2 (5 613 ms against 5 181 ms for a 3 153-token
turn) and the API did not report `cached_tokens`. The harness that *does* report reuse (`engines/naiwa.json` runs report
0.90-0.97 per turn) is the instrument for this, and it currently fails on `doc` turn 1 with a pre-existing OOM (M33
§20.11 recorded the same failure before any M35 change). So the chat comparison is blocked on that harness defect, not on
this milestone.

That is where M35 stands: the mechanism is understood and its 483 ms are reproducible, the design that keeps the
checkpoint without the split is written and its feasibility checked, and the two remaining pieces are (a) the tail-aware
key (§3, which removes the trade) and (b) a harness that can measure the trade in the first place.

## 14. A silent empty answer found while building a harness for this milestone - the loan is no longer the default

The `doc` class had been failing on turn 1 for milestones. §20.11 blamed CUDA OOM, and the harness did die with
`CUDA_ERROR_DEINITIALIZED` plus one `double free or corruption (!prev)`. Both were red herrings: the engine had already
**answered nothing at all**, and the crash was at shutdown.

Reproduced without the harness (same serve command, plain streaming requests):

| prompt | `NAIWA_PREFILL_BORROW=0` | default (loan **on**, since §20.13) |
|---|---|---|
| 33 tokens | 805 ms, real answer | 805 ms, real answer |
| **~65 tokens** | 3 011 ms, real answer | **zero SSE events** |
| ~250 tokens (plain lorem ipsum) | works | **zero SSE events** |
| ~2 000 tokens (a real document) | works | **zero SSE events** |
| 3 154 tokens (two chunks) | works | works |

So: **a prompt that fits in a single streamed chunk returns no token at all when the loan is on.** No error, no NaN
warning, no `finish_reason`, no usage - the stream just closes (the step itself runs: `step_end`, `after_prefix_block`
and `after_reclaim` are all logged, and the prefill rate is normal). Prompts spanning two or more chunks are fine, which
is precisely why §20.13's A/B - every measurement on 2 896- and 3 433-token prompts - saw only the 2x win.

Ruled out by controlled runs, all on the same ~300-token prompt: `NAIWA_PREFILL_NOQUIESCE=1` still empty,
`NAIWA_PREFILL_STREAM_DEPTH=1` still empty, `NAIWA_PREFILL_BORROW=0` fine, `NAIWA_PREFILL_BORROW=1` empty again.
So it is the borrow itself, not the quiesce and not the double buffer. The mechanism is **not** identified: the last
loaned chunk's final hidden state is presumably wrong in a way that makes the first sampled token a stop token,
but nothing checks the hidden states on that path (`NAIWA_BORROW_CHECK` is about non-finite values and none appear).

**Decision: the loan is off by default again** (`NAIWA_PREFILL_BORROW=1` turns it on). A silent empty answer on the
most common prompt shape cannot ship, whatever it costs. The cost is real and is the number the mitigation has to beat:

| 3 143-token prompt | TTFT |
|---|---:|
| loan on (§20.13) | 4 791 ms |
| **loan off (default again)** | **14 083 ms** |

The next concrete step is therefore: **keep the loan, but not when the chunk covers the whole prompt** (a single-chunk
request is the only failing shape). That needs a per-call override of `Slab::borrowing()` - today it is a process-wide
`OnceLock`, and the planner (`plan_with_lend`) is where the "does this chunk cover the whole prompt" fact is known.

Also fixed while here, because it made the `doc` class meaningless: `harness/chat/corpus.jsonl`'s `doc` conversations
contain a `{{DOC}}` placeholder that **run.py never substituted** - the class had been asking the model to summarise the
five characters `{{DOC}}`. `run.py` now takes `--doc-file` (default `harness/chat/doc.md`, ~8 KB of real worklog, small
enough that seven turns stay inside `--pa-context-len 4096`) and substitutes it.

## 16. The loan's silent empty answer: two layout defects, both in `borrow_buffers`

§14 left the loan off by default because a single-chunk prompt answered nothing. The cause is found, and it is not one
defect but two, both in how `borrow_buffers` places a layer's three projections inside its range - plus one harness
defect that hid the symptom for a milestone.

**A. The range was too small for what the layout writes into it.** `span` came from the layers' own
`storage_size_in_bytes`, but the placement pads every projection by `MATRIX_ROW_PADDING` elements and aligns to 256
bytes. `NAIWA_LAYER_DBG`'s new `LAYOUT` line printed the two numbers for every layer, and the biggest layer wrote
**1 664 bytes past the end of its range** - into the expert cache's next *slot*, with no error and no invalidation.
Fixed with a 4 MiB margin on `layer_bytes` (three paddings of a few hundred elements, so 4 MiB is generous; it costs
8 MiB of the loan's two ranges).

**B. The three projections were placed back to back, unaligned.** The private arm (`alloc_like`) ends every projection
with `offset = (offset + bytes + 255) & !255`; `borrow_buffers` used `offset += storage.allocated_bytes()`. For the
block-quantised dtypes the padded byte count is not a multiple of 256 - `IQ3_XXS` is 98 bytes per 256 elements, so a
padded projection ends at a multiple of 4 - so the next projection started at an unaligned address. Those kernels load
16 bytes at a time, so they read across the vector boundary and dequantise garbage.

Defect B is the one that produced the symptom, and the evidence is a fingerprint rather than a guess: with the loan on,
the chunk's MoE output was finite for layers 0-36 and non-finite from **layer 37 onwards** - and 37 is the first layer
whose `gate`/`up` are `IQ3_XXS` (`NAIWA_LAYER_DBG` prints each projection's dtype):

| layer | gate / up | down | chunk output |
|---|---|---|---|
| 36 | IQ2_S | Q2_0 | finite |
| **37** | **IQ3_XXS** | **IQ4NL** | **non-finite** |
| 38-47 | IQ3_XXS | IQ4NL | non-finite (upstream) |

The NaN was row-count independent (300, 700, 1 800 rows all started at 37), buffer-count independent (`NAIWA_DEPTH_CAP=1`
too), and survived `NAIWA_REFILL=0` and `NAIWA_BORROW_NOEMPTY=1`, which is what pointed away from the slot bookkeeping
and towards the placement. `NAIWA_BORROW_PRIVATE=1` (the beside-the-cache arm) is itself broken - it ignores the buffer
index, so both stride buffers share one allocation and it NaNs from layer 0 - which is why it could not be used as the
control; the control was the `alloc_like` code path itself, whose alignment is what B restores.

Measurements after both fixes, loan **on** (`NAIWA_PREFILL_BORROW=1`, or just the new default):

| prompt | TTFT | MoE NaN |
|---|---:|---:|
| ~300 tokens | 3 125 ms, real answer | 0 |
| ~700 tokens | 2 902 ms | 0 |
| ~1 800 tokens | 3 638 ms | 0 |
| 3 143 tokens | 5 742 ms | 0 |
| ~300 tokens, loan **off** | 3 001 ms | 0 |

(That 5 742 ms is this probe harness, not §20.13's; §20.13's own pair, same session, was 4 791 ms with the loan against
14 083 ms without.)

**The chat harness's `doc` class runs end to end for the first time**: 15 turns recorded, none failed. Turn 1 is
1 723 prompt tokens in 3 841 ms, and reuse climbs the way it should - `cached_tokens` 96, 256, 320, 416, 544, 736, 832
for `short`, then 1 792, 1 920, 2 048, 2 176, 2 240 for `doc`.

**So the loan is the default again**, and §14's decision is superseded by this section rather than deleted: it was the
right call while the cause was unknown.

Gates on the shipped defaults: `cargo test` 56 passed, `frac 0` + `--no-lookahead` byte-identical (the oracle arm runs
`--prefix-cache-n 0`, so no cache, no slab, no loan - `borrow_buffers` is unreachable there).

Closing numbers, on the shipped defaults (loan on) after both fixes:

| | |
|---|---|
| `doc` class, 7 turns (1 723 -> 2 264 prompt tokens) | **0 failed**; TTFT 3 528, 1 454, 1 080, 2 651, 819, 2 617, 621 ms |
| reuse across those turns (`cached_tokens`) | 1 792, 1 920, 1 952, 2 080, 2 112, 2 240 |
| a 3 143-token prompt, two runs | TTFT **3 961 / 3 834 ms** with the loan against **14 083 ms** without |

So the loan's 3.6x on a long prompt is back, the short prompt no longer answers nothing, and the checkpoint machinery is
exercised by a corpus for the first time.

## 17. The key does not need to change - and what does need to change is the tail block's retention

Reading the four places the checkpoint touches settles the design, and it is smaller than §3 assumed:

| place | what it does with the key |
|---|---|
| `prefix_cacher.rs:681` `peek_longest_paged_recurrent_prefix` | `filter(key.len() <= max_blocks && block_hashes.starts_with(key))` then `max_by_key(key.len())` |
| `:669` `get_longest_...` | `promote_paged_recurrent_prefix(&block_hashes[..n_blocks], ...)` - a lease over the key's blocks |
| `:488` `add_paged_recurrent_prefix` | `retention.retain(&key)`, `paged_recurrent_sequence_keys[owner] = key` |
| `engine/mod.rs:476` | `max_blocks = cached_tokens / block_size`, restore `n_blocks * block_size` |

Putting the tail hash **inside** the key would make `key.len()` mean "blocks" in three of those and "blocks + 1" in the
fourth, and would feed a non-block hash into `retain`. Keeping the key as block hashes only and storing the tail **beside**
it - `tail: Option<(BlockHash, usize)>`, the hash from the new
`paged_attention::block_hash::tail_hash` and the token count - leaves all four untouched. `tail_hash` is written and its
invariant tested (`cargo test -p mistralrs-core --lib --features cuda tail_hash`; the fork's test suite needs
`PATH=/usr/local/cuda/bin:$PATH`, without it `mistralrs-quant`'s build script fails): it chains like a block hash but
refuses to alias the full block at the same place.

**The blocker, and it is not the key.** The tail's KV rows live in *one partial physical block*, which has no hash - so
`PrefixBlockRetention::retain(&[BlockHash])` (`block_pool.rs:175`, refcounted in `hash_ref_counts`) cannot name it. Once
the sequence that produced the checkpoint releases its blocks, that partial block is ordinary LRU material and another
request can be given it. A restore of `n_blocks * 32 + k` tokens would then read **someone else's KV rows** for the last
`k` tokens: not a miss, silent corruption. So steps 1-3 and 5 below would be *worse* than doing nothing if done without
step 4.

The mechanism for 4 exists: `retain_physical_block(block_id)` (`block_pool.rs:665`) already keeps a block alive by id
(`block_retention_counts`, `num_retained_physical_blocks`), which is how the pool protects blocks that have no hash. The
work is to carry the tail block's id into the checkpoint's lease and to release it where the lease is dropped.

So M35's remaining work, in the order that keeps it correct:

1. ~~`tail_hash`~~ done, with its test.
2. `PagedRecurrentCacheEntry.tail: Option<(BlockHash, usize)>`; `add_paged_recurrent_prefix` takes it; `snapshot_paged_recurrent_prefix` (`pipeline/mod.rs:1931`) stops declining a non-multiple boundary and computes it from `seq.get_toks()`.
3. `peek_longest_paged_recurrent_prefix` takes the prompt's tokens (the caller has them) and, for each candidate entry with a tail, recomputes `tail_hash(block_hashes[n-1], &tokens[n*32 .. n*32+k], extra)` to confirm it; `PagedPrefixCheckpoint` carries the hit's `tail_tokens`.
4. **Retain the tail block by id** in the same lease as the block-hash retention, and release it on eviction. This is the step that decides whether the feature is correct.
5. Relax `engine/mod.rs:465`'s `is_multiple_of(block_size)` gate and make `stage_recurrent_restore`'s `cached_tokens` the hit's `n_blocks * block_size + tail_tokens`; `pipeline/mod.rs:1923`'s `cached_tokens.is_multiple_of` gate on the capture side goes away with step 2.

Steps 2, 3 and 5 are mechanical. Step 4 is the one to do first, because it is what makes the rest safe - and because the
thing it protects (a block with no hash) is the same shape as the failure that made the whole loan look like an OOM in
§14.

## 18. Reuse proven, and it reframes the milestone: the cost is the chunk, not the alignment

A two-turn probe (one process, turn 1 = the 3 143-token prompt, turn 2 = the same prompt plus ten tokens, so the shared
prefix is 3 143) turned up three facts that matter more than the tail key.

**1. The checkpoint is actually used.** `MISTRALRS_PREFIX_DBG=1` shows turn 2 prefilling
`range=Some((3136, 3153))`, i.e. **17 rows** starting at the checkpoint's boundary. This is the first direct evidence in
this repo that the recurrent prefix path works end to end - the chat harness showed it indirectly, through
`cached_tokens` climbing across turns (§16).

**2. But 17 rows cost what 3 143 rows cost.** `NAIWA_STEP_TIME` and `NAIWA_STREAM_DBG`'s `req_time` give the layer loops:
turn 1 (3 143 rows) 3 378 ms, turn 2 (17 rows) **2 386 + 2 599 ms** - and turn 2's TTFT is **7 546 ms** against turn 1's
3 841 ms. The reason is the property M33 §20.13 measured and this milestone kept forgetting: a chunk streams the whole
48-layer expert set regardless of how many rows it has (~45 ms/layer). So "reuse 3 136 of 3 153 tokens" saves rows,
which were never the expensive part.

| arm | turn 1 TTFT | turn 2 TTFT |
|---|---:|---:|
| loan on (default) | 3 841 ms | **7 546 ms** |
| loan off | 21 891 ms | 27 760 ms |
| loan on, `NAIWA_PREFILL_THRESHOLD=1` (17 rows on the GPU stream) | 4 247 ms | 8 827 ms |
| loan on, `NAIWA_NO_TAIL_SPLIT=1` (no tail step, no new checkpoint) | 3 697 ms | 7 092 ms |

The loan is not the problem (turning it off makes both turns 4-5x worse, because the plan chunk drops to 768 and the
prompt needs five chunks instead of two), and neither the CPU path for 17 rows nor forcing that step onto the GPU stream
fixes it (8 827 ms). What costs 7.5 s is **the number of chunks**, each of which re-streams every layer's experts.

**3. That is where M35's value actually is, and it is smaller than §3 assumed.** §12's 483 ms is real, but it is a
*single-shot* saving: it comes from removing one extra chunk that says 7 tokens. For a conversation, the same chunk
boundary already exists (the plan's own chunk end), and the tail key would only save the *last* chunk's alignment - while
the thing that actually decides a follow-up turn's TTFT is that a 17-row continuation is still a whole chunk.

So the recommendation changes shape:

- keep `NAIWA_NO_TAIL_SPLIT` as the knob it is (single-shot prompts: -483 ms, §12) and keep the default truncation for
  chat, where reuse matters;
- **the next milestone-worthy question is not the tail key but "why does a mostly-cache-hit continuation cost a full
  chunk"** - i.e. a 17-row step should be served from the resident experts (the decode path) rather than by streaming
  48 layers. §16's loan is what makes that hard today: the chunk empties the cache it borrows from, so the cheap path
  has nothing resident to serve from.

`tail_hash` (§17) and the partial-block retention plan stay on the shelf: they are correct and needed only if the last
few tokens of a prompt ever matter more than the chunk they force.

### 18.1 Erratum: the loan-off row measured *no reuse*, not the loan's price

The table above labels its second row "loan off", which invites reading 21 891 / 27 760 ms as what the loan costs. It is
not. Re-running that arm with `MISTRALRS_PREFIX_DBG=1` shows the restore declining for a different reason:

```
PREFIX_DBG restore: no_replay_boundary cached=0 replay=NotRequired aux=false
```

`cached=0` is decided *before* the recurrent checkpoint is ever consulted: the KV-level block match found **no retained
blocks at all**, so turn 2 re-prefilled from token 0 in four chunks (`range=Some((0,1024))`, `(1024,1280)`, `(1280,1536)`
...). A checkpoint did exist in that arm ("Recurrent prefix cache full at 1 sequences, 1 checkpoints"), and `capture_gate`
accepted, so what failed is the *retention* of the blocks, not the checkpoint.

That is consistent with the loan protecting them indirectly: the loan returns ~1.3 GiB to the device, so the block pool
sits further from pressure and its retained prefix blocks survive; without it, retained blocks are evicted under
allocation pressure (`RETAINED_PREFIX_PRESSURE_EVICTIONS_METRIC` exists for exactly this). It also means the two arms in
that table differ by **loan *and* reuse**, so the honest reading is:

| arm | reuse | turn 1 | turn 2 |
|---|---|---:|---:|
| loan on | 3 136 of 3 153 tokens | 3 841 ms | 7 546 ms |
| loan off | **none** (blocks evicted) | 21 891 ms | 27 760 ms |

The loan's *own* price is therefore still unmeasured in this probe - and measuring it needs an arm where reuse happens
either way, which is the same "don't borrow for a small step" override the next milestone needs.

## 19. Two corrections, one neutral experiment, and where the real reuse limit is

### 19.1 §18 was wrong about the reuse: that probe never reused

`NAIWA_CHUNK_DBG=1` prints **every** chunk of every prompt step, and turn 2 of the 3 143-token probe is:

```
start=0    end=1536   final=false
start=1536 end=3072   final=false
start=3072 end=3136   final=false
start=3136 end=3153   final=true
```

Four chunks from token 0. The `range=(3136, 3153)` line §18 read as "reuse proven, 17 rows" was the *last* of those
chunks, i.e. the tail of a full re-prefill. So §18's "reuse proven" and §18.1's "retention failed under loan pressure"
are both wrong: **the KV prefix match returned nothing**, which is why the plan started at 0.

### 19.2 The real limit is the retained-block budget

The same probe with a **1 293-token** prompt reuses perfectly:

```
start=1280 end=1303   final=true      # one chunk, 23 rows, from the checkpoint's boundary
```

turn 1 3 123 ms, **turn 2 534 ms**. So the mechanism works, and it works best where it is cheap. What separates the two
probes is the size of the retained prefix: 3 136 tokens is 98 blocks, 1 280 is 40. On this card (`--pa-context-len 4096`,
`--max-seqs 4`, a 2.5 GB expert cache) 98 retained blocks do not survive until the next turn; 40 do. That is a capacity
fact about the block pool, not about the loan, the checkpoint or the alignment - and it is the number a "long-context chat"
feature would have to raise.

### 19.3 "The only chunk does not borrow" is a true statement with no measurable effect

Acting on §18, the prompt path was taught to skip the loan when a step is a sequence's **only** chunk (it starts where the
sequence entered the step and ends at its prompt's end), on the grounds that the loan exists to lower the *number* of
chunks and such a step has no chunk to buy. Implemented and measured, three runs an arm, on the 1 293-token probe
(turn 2 = the 23-row continuation, i.e. exactly the step in question):

| arm | turn 2 TTFT |
|---|---:|
| skip the loan on an only-chunk step | 476 / 520 / 534 ms |
| borrow as before | 563 / 463 / 497 ms |

**Neutral, and the mechanism says why**: `refill_lent` already puts back everything the loan displaced (723 experts, one
queued copy each, then one wait), so the loan's cost to a conversation is that single refill - not the cold cache §18
assumed. What made the 3 143-token probe slow was 19.2's re-prefill, which no borrow policy touches.

The change was therefore **reverted** rather than kept as a knob: it needed a per-step global in mistralrs-core, a
buffer-provenance check (`LayerBuf.borrowed`) so a step cannot reuse the other mode's buffers, and a callback split in
`PrefillStream::decide` between "can borrow" (the plan) and "borrows this step" (the allocation) - and one instructive
failure along the way: sizing the *plan* from the per-step decision dropped the chunk from 3 072 to 1 024 at warmup and
made a 3 143-token prompt cost 9 954 ms instead of 3 841. That lesson is worth keeping even though the code is not:
**a once-per-session decision must not be taken from a per-step fact.**

So M35 ends with: `tail_hash` parked (§17), the truncation knob from §12 shipped as a knob, and the real constraint named -
on this card a conversation can reuse a ~40-block prefix but not a ~98-block one.

## 20. The reuse curve is flat to 3 213 tokens, and one prompt reproduces a total miss

§19.2 read a capacity limit out of one failing probe. Sweeping the prompt length with everything else fixed
(`harness`-identical serve command, two turns, turn 2 = turn 1 plus ten tokens) shows **no cliff**:

| prompt | `cached_tokens` in turn 2 | turn 1 | turn 2 |
|---:|---:|---:|---:|
| 973 | 960 | 2 634 ms | 512 ms |
| 1 453 | 1 440 | 2 909 ms | 439 ms |
| 2 093 | 2 080 | 3 182 ms | 547 ms |
| 2 733 | 2 720 | 3 680 ms | 551 ms |
| **3 213** | **3 200** | 3 926 ms | **541 ms** |
| 3 211 (tail 11) | 3 200 | - | 469 ms |
| 3 215 (tail 7) | 3 200 | - | 533 ms |

So a 100-block prefix reuses as happily as a 40-block one, the tail's length is irrelevant, and **§19.2's "98 blocks do
not survive to the next turn" is retracted** - it was inferred from the one probe that fails, not from a sweep.

What *is* real is narrower and reproducible. `/tmp/pf2900.txt` (a 3 143-token Greek-letter word salad):

| second request | `cached_tokens` | turn 2 |
|---|---:|---:|
| **identical** to the first | **3 136** | **160 ms** |
| first + `"\n\nSay hi."` | **0** | 7 354 ms |
| first + `"\n\nNow summarise the above in one sentence."` | **0** | 7 513 ms |

while the lorem prompts at the same length reuse in both cases (3 200 of 3 213). So appending *anything* to that
particular prefix invalidates it end to end - a total miss, not a partial one - and the identical request reuses it
perfectly. Content-dependent, size-independent, tail-independent, loan-independent (the sweep runs the shipped default).

That is a specific, falsifiable lead rather than a theory: **the next probe is to bisect `pf2900.txt`** (half, quarter,
...) until the property appears or disappears, and to print `PREFIX_DBG`'s block match for both halves. A total miss on an
appended request whose first block is byte-identical to a cached one is not something the block model explains, so either
the hashing of that request's tokens differs from its own earlier hashing - a tokenizer/template path that depends on what
follows the prompt - or the retained entry is dropped for a reason that has nothing to do with the blocks.

Until that is understood, the milestone's own conclusion stands unchanged: the truncation knob is a single-shot
optimisation (§12), `tail_hash` stays parked (§17), and the reuse path is in better shape than the last two sections
claimed.

### 20.1 The miss is one prompt's, and bisection narrows it to its full length only

Same serve command, same probe, second request = first plus ten tokens, unless stated:

| prefix | prompt tokens | turn 1's chunks | `cached_tokens` in turn 2 | turn 2 |
|---|---:|---|---:|---:|
| lorem | 3 213 | `(0,3200) (3200,3213)` | **3 200** | 541 ms |
| lorem (tail 11) | 3 211 | - | 3 200 | 469 ms |
| lorem (tail 7) | 3 215 | - | 3 200 | 533 ms |
| pf2900 / 4 | 790 | - | 768 | 703 ms |
| pf2900 / 2 | 1 579 | - | 1 568 | 466 ms |
| **pf2900** | **3 143** | `(0,3136) (3136,3143)` | **0** | 7 513 ms |
| pf2900, second request **identical** | 3 143 | same | **3 136** | **160 ms** |
| pf2900 + `" "` before the suffix | 3 153 | same | 0 | 7 500 ms |

So the two structurally identical cases - lorem's `(0,3200)+(3200,3213)` and pf2900's `(0,3136)+(3136,3143)` - behave
oppositely, and the same prompt that misses on an appended request hits perfectly on an identical one. Ruled out by the
matrix: prompt length (3 213 works), its tail's length (7, 11, 13 all work), the chunk structure (identical), the block
count (100 works), the loan (all of these run the shipped default), boundary tokenisation (inserting a space changes
nothing), and the prefix's own content in general (the same file at half and quarter length reuses).

What is left is something about *that request pair* - a 3 143-token prompt whose first chunk is the whole prompt minus a
7-token tail, followed by a request that extends it. The next step is to dump the block hashes of both requests
(`compute_block_hashes` is pure, so this is a print, not a debugger) and their pool lookups: if block 0's hash differs,
the pipeline is feeding different tokens; if it is equal, the retained entry was dropped and the metric
`RETAINED_PREFIX_PRESSURE_EVICTIONS_METRIC` will say so. Both are one run each, and neither needs the rest of M35.

## 21. Where naiwa stands against strata, measured now

Same session, same 3 143-token cold prompt, arms interleaved (naiwa, strata, naiwa, strata) so host drift hits both:

| arm | TTFT | whole request |
|---|---:|---:|
| naiwa, shipped defaults | 3 853 / 3 859 ms (median **3 856**) | 4 294 / 4 298 ms |
| naiwa + `NAIWA_NO_TAIL_SPLIT=1` | 3 559 / 3 547 ms (median **3 553**) | 4 139 / 4 105 ms |
| **strata** | 3 205 / 3 192 ms (median **3 198**) | 3 499 / 3 515 ms |

So naiwa is **+20.6%** on TTFT as shipped, **+11.1%** with the truncation knob, and +22.0% / +17.7% on the whole request.

Where the gap is, from the parts already measured:

| part | naiwa | strata | note |
|---|---:|---:|---|
| inside the layer loop (prefill proper) | ~3 179 ms | ~3 198 ms | M33 §22, same prompt: **parity** |
| the 7-token tail chunk | ~300 ms | none | this is M35's premise: strata checkpoints at the prompt's real end, so it never needs the tail step |
| layer-external overhead (A + C segments) | 171 ms (of which C 396 ms with `-n 1`, 112 ms with `-n 0`) | 71 ms | M33 §21/§22 |

The two engine-side numbers that do not depend on the host's memory pressure (nsys, same session, M33 §24) say the same
thing in another currency: **H2D 41.1 GB in 19 948 copies against strata's 41.4 GB in 18 707** - we move slightly fewer
bytes - while kernels are **2.28 s in 23 275 launches against 1.97 s in 14 933**, i.e. 16% more kernel time in 56% more
launches. That capture predates §16's loan fixes, so it is the number to re-take next.

Multi-turn (chat harness, `doc` class, 7 turns, now working for the first time - §16): 0 failures, turn 1 3 528 ms for
1 723 prompt tokens, later turns 621-2 651 ms with 1 792-2 240 cached tokens, TPOT 36-44 ms. Strata's arm of that class has
not been run yet, because the corpus's `{{DOC}}` placeholder was only substituted in §16.

**Caveat that matters**: the host degraded during this session - the *same* naiwa arm measured 3 853 ms in the interleaved
window and 5 825 ms twenty minutes later (the model is 54 GiB in a 60 GiB machine, and each run loads it again). Every
number above holds *within* its window; the cross-window-comparable metric is kernel time.
