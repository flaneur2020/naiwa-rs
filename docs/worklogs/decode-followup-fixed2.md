# Decode follow-up: fixed MTP depth and redundant histogram read

**Status:** investigation; diagnostic patch uncommitted, not a proven performance improvement.
**Parent:** M36 / M38.

## Measurement matrix

All decode requests generate 128 tokens. MTP arms explicitly use `--mtp-n-predict 2`.
Wall timings below are unprofiled, three rounds each; failures were not retried.

| arm | TPOT ms/token | median |
|---|---|---:|
| q8 cache off | 42.4 / 40.7 / 41.2 | 41.2 |
| q8 cache on | 41.6 / 42.4 / 41.7 | 41.7 |
| naiwa, no MTP | 49.4 / 48.5 / 48.5 | 48.5 |
| paired naiwa, q8 off | 44.0 / 40.6 / 41.1 | 41.1 |
| paired Strata | 38.3 / 38.4 / 38.5 | 38.4 |

Strata logs report `drafts accepted 0 of 0`. The paired runs alternate naiwa then Strata.
The fixed-depth q8 A/B does not reproduce a wall-time advantage for caching.

## Output head attribution

`builtin_mtp.rs` calls `mtp_draft_logits` once per draft in `0..n_predict`.
Qwen4-Exp `speculative.rs` forwards this through the shared text lm_head unless a separate draft head is configured.
The text model calls `ctx.lm_head` once after its final mixer. Thus depth 2 normally requires two draft-head
calls and one target-head call; the activations differ, so these are not duplicate computations.

In `/tmp/nsysdec-fixed2-{off,on}.sqlite`, most approximate 48-doorbell intervals have three
q6_k calls with gridX=124160 (half the vocabulary size 248320), totaling 3.33 ms per interval.
This matches the code path but is not a per-call stack attribution. Doorbell counts are not guaranteed
one per model layer when a layer has no CPU misses; do not use division by 48 as exact step counts.
The post-startup windows contain 175/181 vocabulary-shaped calls, 194.17/200.81 ms total.
These kernel sums are not proven wall-time savings. The biggest gap separates startup activity and
the request; its removal is a heuristic request-window cut, not an exact first-token decode marker.

## Redundant histogram read

`src/moe/decode.rs` unconditionally copied `topk_ids` to CPU before calling the disabled histogram recorder.
`Route::submit` subsequently reads routing ids and weights again. Diagnostic patch guards the histogram copy
with `NAIWA_EXPERT_HIST`; `NAIWA_FORCE_HIST_READ=1` restores old behaviour for A/B.

Both arms disable the q8 cache and fix MTP depth to 2.

| arm | TPOT ms/token | median |
|---|---|---:|
| old histogram read | 41.5 / 41.6 / 42.0 | 41.6 |
| guarded histogram read | 40.0 / 42.2 / 44.9 | 42.2 |

Single nsys captures: old 43.2, guarded 41.4 ms/token (not performance replicates).
Whole-capture `cuMemcpyDtoHAsync_v2` calls: 12822 -> 9615. These raw totals include startup and
varying request work, so they demonstrate a removed class of copies but not a normalized speedup.
The routing dependency still exists after deleting the earlier copy; waiting can shift to the real routing read.

## Gates and disposition

- Guarded binary: frac 0/no-lookahead oracle first line byte-identical.
- naiwa tests: 56 passed, 4 ignored.
- Chat harness has not been rerun for this diagnostic patch.
- No stable 1 ms gain established; patch is uncommitted and is not declared shipped.
- Layer-tagged CPU compute probe added subsequently; results below. Per-expert residency simulation is still pending.

Evidence: `/tmp/decode-hist-{old,guarded}{1,2,3}.log`, `/tmp/nsysdec-hist-{old,guarded}.{nsys-rep,sqlite}`,
`/tmp/decode-diagnostic2.log`, `/tmp/oracle-hist-guard.txt`, `/tmp/tests-hist-guard.log`.

## Layer-tagged CPU compute, three 128-token runs

Opt-in `NAIWA_CPU_LAYER_TRACE=1` in `src/moe/stages/cpu_miss.rs` measures fused computation,
not input D2H, queue wait, or result upload. Logs are outside the timer, but still perturb scheduling.
Both q8 caching and the histogram-read guard optimization were disabled for these runs
(`NAIWA_NO_Q8CACHE=1 NAIWA_FORCE_HIST_READ=1`), fixed MTP depth 2.
Diagnostic TPOT: 44.5 / 45.1 / 43.2 ms; do not compare these instrumented times as an optimization A/B.

| run | three-row CPU total | top-eight layer share | top-three layer share |
|---|---:|---:|---:|
| 1 | 2890 ms | 29.3% | 13.1% |
| 2 | 2959 ms | 28.7% | 12.6% |
| 3 | 2845 ms | 28.8% | 12.5% |

Layers 0, 22, 38 are the top three in every run; layers 1, 18, 28, 42 also repeatedly rank high.
Layer 0 has roughly 12 CPU pairs/call and 2.4–2.5 ms median compute; layers 22 and 38 about
9 pairs/call and 1.7–1.9 ms. Their aggregate per-distinct-expert costs are roughly 193–212 us,
so demand size, not a uniquely slow quant kernel, explains much of their higher cost.

Layer 14 is a separate transient: one single-row call takes 140349 / 184989 us in runs 1/2;
run 3 has no comparable spike. Three-row layer-14 costs rank only 18th/20th/22nd.
**Boundary correction:** both large calls are BEFORE `Server listening`, during startup warmup.
They are outside this benchmark request's TTFT and TPOT, not decode-tail optimization opportunities.
The subsequent request's first single-row layer-14 calls are 419 / 403 / 580 us.
Across request-only calls with rows <=3, maximum CPU time is 3572 / 6015 / 4279 us;
p99 is 2752 / 2699 / 2830 us. These are call-time statistics, not token latency.
Root cause inside startup (faults, initialization, scheduling) remains unmeasured;
no claim of page faults or a specific initialization cost is supported.
The benchmark polls the model endpoint before submitting its request; `serve.rs` logs listening
only after model build and warmup. Thus the startup spikes cannot explain measured request TPOT.

Implication: a same-capacity cache allocation study should examine layers 0/22/38 first,
but the top eight account for only ~29% of measured CPU compute. This is not a single-layer bottleneck,
and CPU compute savings must still be checked against exposed GPU wait before converting to TPOT.
No expert-ID trace or cost-aware residency simulation has been collected yet.

Tests after probe addition: 56 passed, 4 ignored. Full chat/oracle have not been rerun for this latest
probe addition. Evidence: `/tmp/decode-cpu-layer{1,2,3}.log`, `/tmp/tests-cpu-layer.log`.

## Expert IDs: demand is mostly diffuse, with an admission-path bias

Extended the same opt-in probe with `ROUTE_IDS` (demand and resident set) and `CPU_IDS` (post-staging CPU pairs).
Three 128-token runs, fixed depth 2, q8 off and legacy histogram read restored. Diagnostic TPOT
49.8 / 45.9 / 48.7 ms; verbose tracing is not a performance arm.

Layers 0/22/38 each have 25 resident slots. Their five most common CPU misses cover only
18–21%, 14–15%, and 13–14% of that layer's CPU pairs respectively. Layer 0 experts 65/107/200
and layer 22 experts 103/22 repeatedly appear among misses, often with near-zero residency.

Selecting five experts per layer from run 1, then evaluating that fixed list:

| run | all three-row CPU pairs | covered pairs | share |
|---|---:|---:|---:|
| 1 (selection set) | 13742 | 324 | 2.36% |
| 2 | 12131 | 248 | 2.04% |
| 3 | 13250 | 252 | 1.90% |

This is optimistic additive coverage, not a same-capacity simulation: it ignores evictions and
changes to staging choice. Runs repeat the same prompt, so this is repeatability, not workload generalization.
Do not infer TPOT savings from these pair counts.

Code explains a potential admission bias: `streamed_misses` ranks by summed router weight,
whereas `promote_staged` can only admit experts already staged. A frequent low-router-weight miss
may remain on CPU despite many sightings. Router weight does not make its exact expert computation cheaper.
This is a testable mechanism, not a proven explanation for individual IDs (weights were not traced).
No cache-policy changes made. Latest tests: 56 passed, 4 ignored; oracle/chat not rerun for ID probe.
Evidence: `/tmp/decode-route-ids{1,2,3}.log`, `/tmp/tests-route-ids.log`.

## Admission ranking measured (three further 128-token runs)

Added opt-in `NAIWA_ADMISSION_TRACE` at staging selection: ranked summed router weights and
chosen IDs. Enabled existing `NAIWA_PROMOTE_TRACE` too. Same fixed-depth, q8-off, legacy-histogram
configuration. Instrumented TPOT 50.8 / 47.0 / 49.9 ms is not an optimization comparison.

For three-row verification calls:

| layer/expert | miss calls, runs 1/2/3 | staged calls | median rank | median staging cutoff |
|---|---|---|---|---|
| 0/200 | 32 / 29 / 31 | 2 / 3 / 2 | 16 / 14 / 15 | 8 / 8 / 8 |
| 0/65 | 42 / 25 / 41 | 3 / 4 / 3 | 14.5 / 13 / 14 | 8.5 / 8 / 8 |
| 22/22 | 23 / 21 / 24 | 0 / 1 / 0 | 12 / 10 / 11 | 7 / 5 / 6 |
| 22/103 | 22 / 25 / 23 | 2 / 3 / 3 | 12.5 / 11 / 12 | 6.5 / 6 / 6 |

Thus low weight ranking really excludes repeatedly requested experts from staging, and hence
from staged-only promotion. This establishes selection bias, not a performance bug: any new resident
requires displacing another one, and changing staging changes CPU/GPU workload balance.

Across all layers, non-staged distinct miss occurrences: 12657 / 11285 / 12108.
Expert IDs missing without staging at least ten times account for 905 / 586 / 658 occurrences
(7.2% / 5.2% / 5.4%). This uses future information from each full run; not a causal policy evaluation.
Broad miss demand remains dominant, so no claim of a 1 ms improvement follows.

Next valid experiment: causal, same-byte-budget replay comparing candidate admission with victim losses,
then one switchable policy only if the replay supports it. No cache policy edited yet.
Tests: 56 passed, 4 ignored. Full chat/oracle not rerun after this latest trace addition.
Evidence: `/tmp/decode-admission{1,2,3}.log`, `/tmp/tests-admission.log`.

## Promotion order: tested, rejected

`dma_miss.rs` builds `staged_now` by iterating `slot_of`, a HashMap. With promotion cap 1,
this makes admission depend on hash traversal rather than weight or frequency order.
`harness/m38/admission_replay.py` compares one-step candidate choices on observed staged sets;
it is not full-cache replay. Past/current demand selects the most frequent eligible staged expert.
Only unambiguous single additions to the next resident set are scored; lookahead may still confound attribution.

| trace | scored decisions | actual next-call pairs reused | frequency next-call pairs reused |
|---|---:|---:|---:|
| 1 | 358 | 215 | 267 |
| 2 | 247 | 148 | 173 |
| 3 | 350 | 201 | 251 |

Most decisions were ambiguous/excluded, so this is only a signal worth testing, not a policy win.
Implemented an opt-in `NAIWA_PROMOTE_FREQUENCY` sorting the existing staged candidate list by
cumulative demand, leaving staging, eviction, memory budget, and promotion cap unchanged.
No verbose probes during wall tests; MTP depth 2, q8 off, legacy histogram read, 128 tokens each:

| arm | three paired TPOT runs | median |
|---|---|---:|
| baseline hash order | 43.3 / 40.8 / 40.6 | 40.8 ms |
| frequency order | 42.9 / 41.1 / 43.2 | 42.9 ms |

Two of three pairs regress; the simple frequency ordering does not deliver the desired improvement.
The experimental policy and switch were removed, not shipped. This does not disprove every admission policy,
but rules out adopting this one on current evidence. Earlier low-weight misses remain outside its candidate set.
Evidence: `/tmp/decode-promote-{baseline,frequency}{1,2,3}.log`.

## Per-step MTP attribution

Fork opt-in `NAIWA_MTP_COST` logs proposer host-call duration and accepted draft count.
Together with `NAIWA_STEP_TIME`, it extends begin/end timestamps beyond the original six-step limit.
No CUDA synchronization added. Step duration includes target execution, verification/commit and
usually proposal generation for the NEXT step; it is not isolated GPU verification time.

Three 128-token runs, depth 2, q8 off, legacy histogram read, no layer/ID logs:

| run | TPOT | verification steps | accepted drafts | zero-accepted steps | step span sum | proposal host span sum |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 40.1 ms | 56 | 70 | 14 (25%) | 5033.1 ms | 215.8 ms |
| 2 | 41.7 ms | 58 | 69 | 16 (28%) | 5238.2 ms | 224.9 ms |
| 3 | 42.9 ms | 60 | 67 | 18 (30%) | 5387.7 ms | 229.7 ms |

Median full-step duration for accepted counts 0/1/2: 89.9/89.5/89.5 ms (run 1),
90.9/86.9/95.0 (run 2), 91.1/87.9/90.0 (run 3). Cost is roughly fixed per three-row verify,
while productive yield differs. Exact output counts need boundary/EOS handling; accepted+steps
is not assumed equal to all 128 emitted tokens.
Zero-accepted steps consume 1240.9/1456.8/1642.6 ms in total, but this is NOT removable cost:
a non-speculative step still computes a target token, and rejection cannot be known in advance.
P(next zero | previous zero) is 29/31/33%, against overall zero rates 25/28/30%; previous rejection
alone is weak evidence for an adaptive policy. These three runs repeat one prompt, not a corpus.

At fixed ~90 ms/step and ~2.2 tokens/step, improving output yield by ~0.05–0.06 tokens/step would
approximately save 1 ms/token if step cost stayed fixed. This is a sensitivity calculation, not
an achieved optimization. Direct proposal host-call spans sum to ~1.7–1.8 ms/output token, but
contain asynchronous work and are not a complete isolated GPU drafting budget.

Evidence: `/tmp/decode-mtp-fullcost{1,2,3}.log`. Probes remain uncommitted; full oracle/chat gates
have not been rerun for these fork-only timing additions. No performance mechanism changed.

## Cross-run text consistency and dynamic-cache isolation

`/tmp/decodebench.py` now optionally saves content chunks, full text, usage, hash and timing to
`DECODE_CAPTURE`. These are text-byte comparisons, not raw token-ID comparisons; TPOT uses
reported completion_tokens=128, not content chunk count (Strata emits 125 content chunks).

| arm | three TPOTs ms | text hashes (prefix) |
|---|---|---|
| naiwa MTP=2 | 42.4 / 44.3 / 41.4 | 4a54db4d / b6f88441 / c651bc5c |
| interleaved Strata | 38.2 / 38.3 / 38.5 | daaedd64 / daaedd64 / daaedd64 |
| naiwa no MTP | 48.8 / 49.0 / 49.4 | daaedd64 / 5d353658 / 4a54db4d |
| naiwa no MTP, no lookahead, promotion cap 0 | 55.0 / 54.4 / 54.7 | 499135c6 / 499135c6 / 499135c6 |

All naiwa arms: q8 cache disabled, legacy histogram read restored, frac 0.4, 128-token
chat requests (NOT oracle). MTP arm includes lightweight cost logs. First two arms interleaved;
subsequent ablations sequential, not a controlled cross-arm speed comparison.
MTP verify counts 58/63/57, accepted draft sums 69/63/69. Text differences start at character
87 and 204 relative to first MTP run; no-MTP differences at character 49 and 299.

Findings: MTP is not required for text divergence. Removing both lookahead and promotion yields
byte-identical text across these three repetitions, supporting dynamic residency as a suspect.
This combined ablation does NOT distinguish lookahead from promotion, nor prove a universal
numerical stability guarantee. The frozen arm is slower and is diagnostic, not recommended.
Possible causal chain: changing residency -> differing CPU/GPU arithmetic -> greedy divergence
-> different MTP acceptance/work counts. Residency and logits at the first divergence still need
comparison to establish this chain. CPU/GPU numerical differences are not by themselves proof
of incorrect outputs.
Evidence: `/tmp/consistency-{naiwa,strata,nomtp,frozen}{1,2,3}.json` and corresponding
`/tmp/decode-consistency-*.log`. No new engine code was changed for this isolation matrix.

## Separate lookahead/promotion and deterministic-order ablation

No-MTP, frac 0.4 chat requests, 128 tokens, q8 off and legacy histogram read, three runs per arm:

| arm | TPOT ms | text hash prefixes |
|---|---|---|
| lookahead off, promotion on | 51.2 / 50.4 / 50.6 | c651bc5c / e19ca091 / e19ca091 |
| promotion off, lookahead on | 49.7 / 50.1 / 49.6 | b1eeb110 / a90d48dd / e19ca091 |
| lookahead off, promotion sorted by expert ID | 50.8 / 50.4 / 52.8 | e19ca091 / c651bc5c / 1349fe10 |

First two arms interleaved. Fixed-ID ordering was a separate opt-in implementation, then removed:
it did NOT restore text determinism. Thus hash traversal is not a sufficient explanation for all divergence.
Individual lookahead/promotion ablations also do not isolate a sole culprit. Combined disable remained
consistent in the earlier three-run sample; that is evidence, not a universal guarantee.
Need compare initial resident sets and first resident/logit divergence under fixed-order promotion;
startup and other mutation/timing paths have not been ruled out. Do not claim a demonstrated race or
numerical correctness bug from these text differences alone.
Evidence: `/tmp/consistency-{nolook,noprom,idorder}{1,2,3}.json` and corresponding service logs.








## Root cause: nondeterministic expert aggregation and promotion order

Two mechanisms made repeated runs of one greedy request take different token paths. Both are absent
from Strata, whose decode repeats bit-identically (script `harness/m38` evidence in the table above).

**1. GPU expert combine used `atomicAdd`.** `mistralrs-quant/kernels/indexed_moe/indexed_moe.cu`
accumulated a row's top-k contributions with `atomicAdd(&out[row], tmp * tw)`, so the order of the
ten adds follows block scheduling. Fork path behind `MISTRALRS_MOE_DET_COMBINE=1`: when the weight
pointer is null the down kernel writes each slot's unweighted partial to its own row, and a new
`moe_combine_topk_weights` kernel applies the weights and sums slots in ascending order. One extra
launch per GPU MoE call, partials reuse a static F32 workspace (batch*topk*hidden, 100 KB at decode).

**2. Promotion candidates came from a `HashMap`.** `dma_miss::stage_misses` builds `slot_of` as a
`HashMap`, `staged_now` inherits that order, and `promote_staged` with `cap = 1` promotes whichever
eligible candidate is seen first. Hash order differs per process, so *which* expert became resident
differed between runs; that changes the CPU/GPU placement of later steps and therefore their rounding.
Fix: sort `staged` by expert id before the cap is spent (`src/moe/cached_layer.rs`). Cost: sorting at
most a few dozen entries per layer; `NAIWA_PROMOTE_FREQUENCY` was a different, previously reverted
experiment.

Measured, no MTP, frac 0.4, q8 off, `--no-lookahead`, three runs per arm:

| arm | TPOT ms | text hashes |
|---|---|---|
| atomic combine, hash-order promotion | 51.2 / 50.4 / 50.6 | c651bc5c / e19ca091 / e19ca091 |
| deterministic combine, hash-order promotion | 51.4 / 51.1 / 50.7 | 4a54db4d / 32890df3 / e19ca091 |
| deterministic combine, id-order promotion | 50.3 / 49.8 / 50.5 | 320a8861 / 320a8861 / 320a8861 |
| deterministic combine, id-order, frac 0 | 74.8 / 74.1 / 74.1 | b6f88441 / b6f88441 / b6f88441 |

Fixing only the combine is not enough; fixing only the promotion order was not enough either (earlier
arm, with the atomic combine still active). Both were required for this prompt.

`NAIWA_XSUM` (MoE-input checksum, diagnostic only, first three ticks) localized the divergence when
both fixes were absent: ticks 1 and 2 were bit-identical at all 48 layers, the first difference was
tick 3 layer 2, and the `NAIWA_STAGE_TRACE` staged set at tick 3 layer 1 differed between runs
(`{33,95}` vs `{11,33,95}`). That is residency/placement before numerics, not a difference in the
request.

### Still nondeterministic: the MTP drafter

With both fixes, MTP=2 still produced three different texts (58/57/59 verify steps, 69/69/68 accepted
drafts, TPOT 41.7/41.0/42.2) against 42.9/41.0/41.4 with the deterministic combine off. The drafter's
resident head is a `mistralrs_core::MoEExperts` (`src/moe/resident.rs`), whose grouped kernels
(`kernels/moe_grouped/moe_grouped.cu`) also aggregate with `atomicAdd` across top-k. Making that path
deterministic is the remaining step; until then the MTP arm's work counts and TPOT can still vary.

Cost of the deterministic combine at MTP=2: median 41.7 vs 41.4 ms (n=3 each, not interleaved, inside
this machine's spread) — no measurable regression, no demonstrated speedup.

Oracle (frac 0, `--no-lookahead`) first line byte-identical with id-order promotion. Full test suite and
chat harness were NOT rerun for these changes. `MISTRALRS_MOE_DET_COMBINE` is opt-in; the promotion sort
is compiled in.
