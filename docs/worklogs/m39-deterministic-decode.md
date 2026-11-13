# M39 — the same request produces the same tokens (reproducible decode)

**Status:** 🔶 in progress (2026-11-11) — two of three nondeterminism sources fixed and gated; the
drafter's expert combine is still atomic. **Parent:** M38 follow-up
([`decode-followup-fixed2.md`](decode-followup-fixed2.md), where the divergence was found).

M36–M38 chased decode cost per token. This milestone closes the other half of the same question: **is
a repeated run doing the same work?** It was not, and that made every earlier A/B suspect.

## 1. The observation

Same engine, same greedy request (`temperature 0`, `top_p 1`, `seed 0`, 128 tokens, one fixed prompt),
three consecutive runs. Text is compared as the concatenated streamed content, and the response's
`completion_tokens` is used for TPOT (not the chunk count — Strata emits 125 content chunks for 128
tokens).

| engine / configuration | three TPOTs (ms) | three output texts |
|---|---:|---|
| Strata (`--spec 4`, cache 800, no drafts accepted) | 38.2 / 38.3 / 38.5 | **identical**, 3/3 |
| naiwa, `--mtp-n-predict 2` | 42.4 / 44.3 / 41.4 | **three different** |
| naiwa, no MTP | 48.8 / 49.0 / 49.4 | **three different** |

Strata's counters agree with its text: repeated requests report the same
`26485 hits / 51744 lookups / 9696 PCIe` in `/tmp/decode-strata-engine.log`. Its TPOT is stable
because it repeats the same work, not because its per-kernel timing is less noisy.

MTP is an *amplifier*, not the origin: with drafting, a changed token changes acceptance, so the
number of verification steps moves too (58 / 63 / 57 steps, 69 / 63 / 69 accepted drafts for
42.4 / 44.3 / 41.4 ms). Without drafting the text still diverges, but TPOT spread collapses to
0.6 ms — the same request then costs nearly the same even with different tokens.

## 2. Ablations: it is not one switch

No MTP, `--pcie-frac 0.4`, q8 cache off, 128 tokens, three runs per arm:

| arm | TPOT ms | texts |
|---|---|---|
| lookahead off, promotion on | 51.2 / 50.4 / 50.6 | 2 distinct |
| promotion off, lookahead on | 49.7 / 50.1 / 49.6 | 3 distinct |
| lookahead off **and** promotion off | 55.0 / 54.4 / 54.7 | identical |
| lookahead off, promotion candidates sorted by id (atomic combine still active) | 50.8 / 50.4 / 52.8 | 3 distinct |
| everything off (`--expert-cache-gb 0.01`) | 87.9 / 88.8 / 88.2 | identical |

Removing dynamic residency made the run reproducible, and so did removing GPU expert compute
entirely. That pointed at *where experts execute*, but not yet at *why* that differs.

## 3. Source 1 — the GPU expert combine is an `atomicAdd`

`mistralrs-quant/kernels/indexed_moe/indexed_moe.cu`, fused down+aggregate:

```cuda
atomicAdd(&out[row], tmp * tw);   // one add per (row, top-k slot), any order
```

Ten blocks per row add into the same address; the order follows block scheduling, so the f32 sum
differs in the last bits between runs. Strata's equivalent (`src/kernels/cuda/native_moe.cu`) instead
writes per-expert partials and reduces them in a fixed loop (`for expert = 1..k`), which is why its
repeated output is bit-stable.

Fork fix, behind `MISTRALRS_MOE_DET_COMBINE=1`: a null weight pointer makes the down kernel write each
slot's unweighted partial to its own row, and a new `moe_combine_topk_weights` kernel applies the
weights and accumulates slots in ascending order. One extra launch per GPU MoE call; the partials
reuse a static F32 workspace (100 KB at decode).

## 4. Source 2 — promotion candidates come out of a `HashMap`

`dma_miss::stage_misses` builds `slot_of: HashMap<u32, …>`; `staged_now` inherits that iteration
order, and `promote_staged` with the default `cap = 1` promotes whichever eligible candidate it sees
first. Hash order is randomized per process, so *which* expert became resident differed between runs.
A different resident set means a different CPU/GPU split for later steps, hence different rounding.

Fix (`src/moe/cached_layer.rs`): sort `staged` by expert id before the cap is spent. A few dozen
entries per layer, well under a microsecond.

Neither fix alone was sufficient — with only the combine fixed, or only the promotion order fixed, the
three runs still diverged:

| combine | promotion order | TPOT ms | texts |
|---|---|---|---|
| atomic | hash | 51.2 / 50.4 / 50.6 | 2 distinct |
| deterministic | hash | 51.4 / 51.1 / 50.7 | 3 distinct |
| deterministic | expert id | **50.3 / 49.8 / 50.5** | **identical, 3/3** |
| deterministic | expert id, `--pcie-frac 0` | 74.8 / 74.1 / 74.1 | identical, 3/3 |

## 5. Where the first difference actually appears

`NAIWA_XSUM` (diagnostic: checksum of each layer's MoE input, first three ticks) removes the guesswork.
With both fixes absent: ticks 1 and 2 are bit-identical at all 48 layers, the first difference is
**tick 3, layer 2**, and `NAIWA_STAGE_TRACE` shows the staged set at tick 3 layer 1 differed between
runs (`{33,95}` vs `{11,33,95}`). Order of events is therefore *placement first, arithmetic second* —
the model was not fed different data; a different expert ran on a different device.

`NAIWA_ARGMAX_DBG` (fork sampler, top-1/top-2 margin, host-side only — the logits vector is already
copied for the argmax) explains why a last-bit difference becomes a different token: the logits arrive
**bf16-quantized**, so margins are multiples of 0.125 around 20. The first token divergence of that
sample was at a step whose margin was **0.125 in one run and 0.0 in the other**. In this regime a
1-ulp perturbation decides the token.

## 6. Still nondeterministic: the drafter

With both fixes, `--mtp-n-predict 2` still produced three different texts (and 58 / 57 / 59
verification steps). The drafter's resident head is a `mistralrs_core::MoEExperts`
(`src/moe/resident.rs`), and its grouped kernels
(`kernels/moe_grouped/moe_grouped.cu`) aggregate across top-k with `atomicAdd` as well. Until that path
is fixed, the MTP arm's work counts — and therefore its TPOT — can still vary run to run.

## 7. Acceptance criteria

Consistency is the milestone's product, so the gates are reproducibility gates, not timings:

1. **Text determinism, no MTP:** 3 consecutive runs of the fixed prompt at `--pcie-frac 0.4`,
   128 tokens, produce byte-identical text.
2. **Text determinism, MTP=2:** same, with drafting on, *and* identical verification-step and
   accepted-draft counts. (Currently failing: §6.)
3. **Demand determinism:** the per-layer routed-id sequence is identical across those runs
   (`NAIWA_ROUTE_IDS`), which is stricter than the text comparison and catches divergence that has not
   yet reached a token.
4. **Oracle:** `frac 0` + `--no-lookahead` stays byte-identical to `harness/m1/reference.oracle.txt`.
   Verified for the promotion sort; must be re-verified once the drafter path changes.
5. **Regression:** 56 unit tests and the 154-turn chat harness, no empty replies.
6. **Cost:** the deterministic combine must not cost wall clock. Measured so far: MTP=2 median
   41.7 ms with it against 41.4 ms without (three runs each, not interleaved, inside this host's
   spread). Not a gate — a check that the fix is free.

Explicitly **not** gates: TPOT medians, launch counts, hit rates. They are recorded, and they cannot
become the justification for keeping a mechanism.

## 8. Work items

| # | item | state |
|---|---|---|
| 1 | deterministic down+aggregate combine in the fork (`MISTRALRS_MOE_DET_COMBINE`) | built, opt-in |
| 2 | promotion candidates sorted by expert id | built, compiled in |
| 3 | deterministic aggregation in the drafter (`moe_grouped.cu`) | **open** |
| 4 | interleaved cost A/B for (1) at MTP=2, and decide whether it becomes the default | **open** |
| 5 | same three-run check on a second prompt and on `frac 0` | **open** |
| 6 | 56 tests + 154-turn chat after (3) | **open** |
| 7 | per-step split of the 10–11 ms gap (§11): attribute it to exposed CPU expert time against GPU chain, one instrument per arm | **open** |
| 8 | rewrite the roadmap's "`frac > 0` is not byte-exact" note: the cause is now identified, so the
      statement should say *which* nondeterminism remains, not "near-ties flip" | **open** |

## 9. What this milestone is not

- Not a correctness claim. Different orderings of the same sums are not a bug, and a changed token is
  not evidence of a wrong token. What is claimed is narrower: **the same run did different work.**
- Not a performance milestone. Determinism does not make decode faster; it makes 1 ms claims
  measurable. The 2–3 ms gap to Strata is still a cost question (M36/M38).
- Not a reason to freeze the cache. The reproducible configuration at `--pcie-frac 0` is 74 ms/token,
  and the "everything off" arm is 55 ms — both are diagnostics, not candidates.

## 10. Evidence

- Texts and usage: `/tmp/consistency-{naiwa,strata,nomtp,frozen,nolook,noprom,idorder,argmax,routeids,frac0-*,det*,sortp*,mtpfix*,mtpa*}{1,2,3}.json`
- Service logs: the matching `/tmp/decode-consistency-*.log`; MTP step counts `/tmp/decode-mtpfix*.log`.
- Strata reference log: `/tmp/decode-strata-engine.log` (counters repeat across requests).
- Probes, all default off: fork `NAIWA_ARGMAX_DBG`; engine `NAIWA_XSUM`, `NAIWA_ROUTE_IDS`,
  `NAIWA_STAGE_TRACE`, `NAIWA_MTP_COST`; fork switch `MISTRALRS_MOE_DET_COMBINE`.
- Prior analysis and the negative-results matrix: [`decode-followup-fixed2.md`](decode-followup-fixed2.md).

## 11. One pure decode step, naiwa against Strata

With drafting removed from the picture on both sides, the two engines can be compared per step. Protocol:
one engine process, three consecutive 128-token greedy requests on the fixed prompt; naiwa without MTP,
`--pcie-frac 0.4`, default cache and lookahead, plus `NAIWA_STEP_TIME`/`NAIWA_STEP_TIME_ALL` so every
pipeline step is timestamped (the instrument is not free, but it is a pass-through `eprintln` at step
boundaries); Strata with `--spec 4 --suffix-draft 0`, which its own log reports as
`drafts accepted 0 of 0` — no draft tokens were ever proposed, so its step is a plain one-token step.

Strata cannot be started with `--spec 0`: the native IQ pack path rejects it
("needs --native SHARD1, --spec T (T >= 2)"). `0 of 0` is therefore the closest available "no drafting".

| engine | decode step | wall per token |
|---|---:|---:|
| **naiwa**, request 1 | median **48.5** ms (min 40.1, p90 57.8, max 69.6) | 49.7 ms |
| **naiwa**, request 2 | median **47.4** ms (min 36.6, p90 57.7, max 65.0) | 49.0 ms |
| **naiwa**, request 3 | 125 steps timed | 48.3 ms |
| **Strata**, request 1 (cold) | **38.5** ms | 38.4 ms |
| **Strata**, request 2 (prompt reused) | **36.7** ms | 36.7 ms |
| **Strata**, request 3 (prompt reused) | **37.0** ms | 37.0 ms |

Over all 381 timed naiwa decode steps: p10 42.5, p50 47.7, p90 57.1, p99 65.0, min 36.6, max 69.6;
**35.4% of steps are 50 ms or slower**. naiwa's step spans account for its whole wall time (the sum over
one request is within one step of `total - ttft`), so there is no hidden per-token host gap to blame.

Consequences:

- The gap is **≈ 10–11 ms per pure decode step** (47.4–48.5 against 36.7–38.5), i.e. 26–30% — and it is a
  *distribution*, not a constant: naiwa's fastest steps (36.6–40.1 ms) are already at Strata's average,
  while a third of its steps take 50 ms or more.
- That matches the earlier component numbers rather than contradicting them: M38's perfect-cache probe
  put naiwa's GPU-only chain at ≈26.4 ms, and the CPU expert batch was ≈19 ms of a 41–45 ms step at the
  time. A step here is 47.7 ms, so roughly 20 ms of CPU expert work is still exposed. Strata also runs
  experts on 6 pool threads (its log: `CPU pool tasks/phase: 18`, 50.8% cache hit, 16.0% over PCIe) and
  lands at 37 ms, which says the difference is **how much of that CPU term is hidden**, not whether it exists.
- MTP hides it: at depth 2 a 3-row step costs ~90 ms (~41 ms/token), where the same CPU work is amortised
  over ~2.2 tokens. Per-row the GPU chain barely grows (49 ms for 1 row, 90 ms for 3), so drafting is
  currently the engine's main defence against the exposed CPU term.

Caveats specific to this table:

- Strata's pack is a native re-quantisation of the same model, not the same bytes as the IQ1_M GGUF, so
  per-byte costs are not identical. Cache sizes also differ (800 slots = 1.98 GiB against 2.5 GB).
- Strata reports per request, not per step, so its spread across steps is unknown; only its average is
  comparable, and the two engines' step definitions are taken from their own logs.
- Different probe overhead on the two sides: naiwa's per-step `eprintln` is inside the measured engine.
- One prompt, one host state, three requests each. These numbers are a magnitude comparison, not a
  calibrated A/B.

Evidence: `/tmp/pure-pure-naiwa3.log` (step timestamps), `/tmp/pure-pure-naiwa3-{1,2,3}.json`,
`/tmp/pure-strata-pure.log`, `/tmp/decode-strata-pure.log` (engine `generated in` lines), scripts
`/tmp/purestep.sh` and `/tmp/strata-nospec.json`. In-process Strata requests produced three different
texts (see §6 note), so its *average* per-step time is the only quantity used here.

## 12. Where one no-MTP step's time and variance actually go

Measured on the §11 protocol (`--cpu-moe`, 2.5 GB cache, `--pcie-frac 0.4`, no MTP), one engine process,
one 128-token request, with the existing probes turned on one at a time. All probes are in-process and
add jitter, so the totals below are attribution, not TPOT.

### 12.1 PLE: readahead is already off, and there is no disk traffic

The n-gram (PLE) table is a 28 GB memory-mapped GGUF tensor, gathered on the host every step
(`ple.rs`: "out of the page cache when it holds them, off the disk when it does not"), and `PleTable::load`
applies `madvise(MADV_RANDOM)` to the mapping — kernel readahead is disabled already.

`PLE_DBG=1`, 129 gathers in one request: **median 0.1 ms/call, p90 0.1, max 1.1, 11.9 ms in total**
(0.09 ms/step), 19 rows/call.

`/proc/<pid>/io` + `/proc/<pid>/stat` + `/proc/vmstat` sampled every 10 ms across the decode:
`read_bytes` **+0**, `majflt` **+0**, `pgscan_kswapd` **+0**, `pgsteal_direct` 7 pages, minor faults
~16/step, correlation with step duration **r = 0.10**. So the variance is not PLE I/O, and it is not page
reclaim either. Per request the table footprint is tiny (19 rows × 1 320 B × 128 ≈ 3 MB), which is why
nothing misses.

The residual risk is real but latent: `MADV_RANDOM` makes a *miss* a single synchronous 4 KB read with no
readahead, so if the hot region were ever evicted (55 GB RSS against 60 GB RAM) the cost would appear as
rare, large spikes rather than a smooth slowdown. That is untested because no miss occurs.

### 12.2 The step is host-paced, and the routed-expert path is 28 of its 48 ms

`NAIWA_STEP_DBG=1` classifies every layer entry: was the previous layer's GPU work complete?
**busy 0/48 for 100% of windows**, dry 0.97 ms/layer — at every entry the GPU had already finished. The
host is the pacer, and the step is the sum of 48 host-paced layer sections.

`NAIWA_STEP_BUDGET=1`, 128 steps, medians (p10 → p90):

| phase | ms/step | p10 → p90 |
|---|---:|---|
| **total (the 48 routed-expert calls)** | **28.4** | 24.1 → 35.9 |
| plan (top-k doorbell read + hit/miss split) | 21.3 | 17.4 → 27.7 |
| — of which GPU expert kernels (`expert_dev`) | 14.0 | 11.6 → 18.1 |
| staging_wait (PCIe DMA) | 2.5 | 2.2 → 2.8 |
| gpu (queueing the expert kernels) | 2.0 | 1.9 → 2.1 |
| stage (slot reservation + copies) | 1.1 | 0.9 → 1.3 |
| spawn + cpu_join + other | 1.4 | — |

Two facts matter here. First, the phase that carries the spread is `plan` (10.3 ms p10→p90), and inside it
the GPU expert kernels (6.5 ms p10→p90) — the variance lives in the MoE path, not in staging or the host
bookkeeping. Second, **28.4 ms is only the routed-expert path**: the step spans 48 ms, so ~20 ms/step is
spent outside `CachedExperts::forward` — attention/GDN, shared experts, norms, the 248k-vocab head,
sampling, detokenisation. That 20 ms has no instrument on it yet and is the largest un-attributed block.

### 12.3 What the step asks for

`NAIWA_ROUND_STATS=1`, per 4-pass window: **54.7% GPU-resident hits, 22.0% staged over PCIe, 23.3% CPU**;
~200 MiB staged per step (≈ 15 ms at the 13.4 GB/s the PCIe probe measures), ~12.9 ms of CPU gemv per step.
(An earlier revision of this section divided the 4-pass dump by 8 and said 103 MiB; §13's independent
nsys measurement of 225-347 MB moved per token is what corrected it.)
`pairs 480, distinct 480` — every layer's 10 experts are distinct, so there is no per-step grouping headroom.

Step time does **not** track those fractions at 4-step resolution (staged% r = −0.19, cpu% r = −0.17,
staged MiB r = −0.16, n = 32 windows). Either the spread is per-step rather than per-window, or its driver
is not workload volume. That is unresolved and is the reason §8 item 7 stays open.

Evidence: `/tmp/ple-dbg1.log`, `/tmp/ple-dbg1.io`, `/tmp/sam-dbg3.txt`, `/tmp/step-busy1.log`,
`/tmp/step-budget1.log`, `/tmp/step-round1.log`, scripts `/tmp/pledbg.sh`, `/tmp/pledbg2.sh`,
`/tmp/sampler.py`, `/tmp/stepdbg.sh`.

## 13. GPU timeline: the step moves data, it does not do maths

nsys on one 128-token no-MTP decode, same protocol as §11, decode window = the last 6.5 s. Sums are GPU
kernel/copy durations, so they add up over streams and can exceed wall time.

| | naiwa, lookahead on | naiwa, lookahead off |
|---|---:|---:|
| sum of device op durations (over-counts streams) | 55.2 ms/token | 46.0 ms/token |
| **union of device op intervals (real occupancy)** | **45.2** | **45.0** |
| kernels + copies issued | 3 835 /token | 3 581 /token |
| **H2D copies** | **347 MB, 820 copies, 27.4 ms** | **225 MB, 642 copies, 17.8 ms** |
| — of which lookahead prefetch | ~120 MB, ~9.6 ms | — |
| D2D copies | 62 MB | 75 MB |
| D2H copies | 0.7 MB, 136 copies, 0.36 ms | — |
| wall per token | 50.8 ms | 52.6 ms (nsys arm; not a clean A/B) |

Top consumers per token: `[CUDA memcpy Host-to-Device]` **27.2 ms**, `doorbell_wait_copy_f32_kernel`
**9.1 ms**, `mmvq_gguf_q6_k_bf16` 3.9 ms (the LM head), `gemv_kernel_batched` 2.0 ms,
`cutlass wmma 32x32x128 tn_align8` 1.7 ms, then the MoE expert kernels proper
(`moe_gemv_down_aggregate_iq4nl` 0.86, `moe_gemv_fused_gate_up_iq2s` 0.61) — **≈ 2.8 ms in total**.
Average op duration is 14 µs; 3 835 ops per token is the shape of the step.

Two structural facts follow:

1. **The expert weights we stage are the dominant traffic, and they are staged at full PCIe bandwidth.**
   `NAIWA_ROUND_STATS` reports ~200 MiB/step staged; nsys reports 225 MB/token with lookahead off — the
   same number by two independent paths. At 13.4 GB/s that volume *is* 17 ms, so the copies are not
   inefficient, there are simply too many of them. Turning lookahead off removes ~120 MB of speculative
   copies and ~9.6 ms of GPU-side copy time.
2. **Both ends of the layer's round-trip are paid.** `doorbell_wait_copy_f32_kernel` is the *result* leg
   (`src/doorbell/doorbell.cu:105`): the GPU spins on a mapped `seq` until the host has published the CPU
   experts' output, then copies the slot in. It burns **9.1 ms/token** of GPU time waiting for the CPU
   experts, while the host separately spends ~21 ms/step in `plan` (§12.2) waiting for the GPU's top-k and
   expert kernels. Strata's split of the same structure is `waitB` (11.3 ms) and `waitCPU` (7.7 ms) inside
   a 34.7 ms GPU stream that contains only ~15.4 ms of real kernels
   (`STRATA_DECODE_TIMING=1 STRATA_VERIFY_PROFILE=1`).

So the comparison is not "their kernels are faster". Per token naiwa occupies the GPU for **55 ms and
moves ~410 MB** to produce one token; Strata's GPU stream is 34.7 ms of which ~15 ms is arithmetic and
~19 ms is structured waiting. naiwa's step is a data-movement and synchronisation problem.

Caveats: the nsys arms are single runs under a profiler, so their wall times are not a performance A/B
(the `--no-lookahead` arm looks *slower* at 52.6 ms; only the copy volumes and the op counts are used
here). `Bytes (MB)` from nsys is decimal MB, `ROUND_STATS` is MiB.

Evidence: `/tmp/ns-naiwa-trace.csv` (+ `/tmp/nsys-nsnaiwa.nsys-rep`), `/tmp/ns-nolook.csv`,
`/tmp/step-round-nolook.log`, Strata's `/tmp/decode-strata-timing.log` (request 2, warm:
`... CPU 21.04 ... total 34.65 ms/window`, `waitB 9.23 + 2.05`, `waitCPU 6.31 + 1.39`).

## 14. The step, closed: 17 ms maths, 27 ms copies, 9 ms spent waiting on the CPU

Take the decode window as exactly 6.5 s = 128 tokens (span 50.8 ms/token, the bench's own number), merge every
device op interval to get occupancy, and intersect gaps with what precedes them. Per token:

| | lookahead on | lookahead off |
|---|---:|---:|
| **occupancy** (union of all device ops) | **43.7** | 43.3 |
| ... kernels only (no copies, no spin) | **17.1** | 15.9 |
| ... copies only (`memcpy`/`memset`) | **27.1** | 17.6 |
| ... `doorbell_wait_copy_f32` only | **9.4** | 11.3 |
| occupancy without the spin | 38.2 | 32.0 |
| ⇒ spin is additive / overlapped | 5.5 / 3.9 | 11.3 / 0.0 |
| **device idle** (span − occupancy) | **7.1** | 7.5 |

Four things the earlier "55 ms busy" phrasing hid:

1. **Copies and kernels barely overlap**: 17.1 + 27.1 = 44.2 against a 38.2 union without the spin, so only ~6 ms
   of copy time hides behind kernels and **~21 ms of it is exposed**. The copy engine is a stage of the pipeline, not
   a side channel.
2. **The spin is mostly exposed** — 5.5 of 9.4 ms additive with lookahead on, **11.3 of 11.3 additive** with it off.
   That is the device sitting on `doorbell_wait_copy_f32` because the CPU experts have not published yet: the CPU
   expert time is waited for, not overlapped.
3. **That is what lookahead actually buys.** It does not shorten anything; it fills the CPU wait with speculative
   copies (120 MB/token). With it off, copies drop 27.1 → 17.6 and the spin absorbs the difference (9.4 → 11.3),
   so occupancy and wall time are unchanged (43.7 vs 43.3). The two arms swap which stage binds. The earlier nsys
   wall-clock equality is therefore a re-balance, not noise.
4. The step is `max(kernel chain, copy chain) + sync points`, and today the copy chain is the one that binds.

### 14.1 Where the 7.1 ms of idle sits

Gaps ≥10 µs between merged intervals (14 297 of them, 3.89 ms/token), attributed by the op ending at the gap's start
and the op starting at its end:

| idle ms/token | share | after → before |
|---:|---:|---|
| 1.55 | 40% | `D2H copy` → `H2D copy` |
| 0.70 | 18% | `D2H copy` → `D2H copy` |
| 0.42 | 11% | `copy2d_f32` → `copy2d_f32` |
| 0.19 | 5% | `H2D` → `H2D` |
| 0.19 | 5% | `cast_` → `cast_` |
| 0.19 | 5% | `silu` → `silu` |

**58% of the attributable idle follows a D2H copy** — the device with nothing queued while the host consumes the
routing read. That is precisely the window a device-side plan removes, and it is **2.3-3.9 ms/token**, which
reproduces M28 §8's independent measurement (3.97 ms/token = host-blocked ∩ device-idle) on a different trace.

### 14.2 What each candidate is worth, measured

| candidate | ceiling | why |
|---|---:|---|
| device-side plan (M28 §5 residency map + device compaction) | **~2.3-3.9 ms/token** | §14.1: idle after D2H |
| removing the CPU-expert wait | up to **9.4 ms/token** (all of the spin is additive with lookahead off: 11.3) | §14 row 2 |
| fewer PCIe bytes at the same overlap | **0 in this arm** — re-balances into the spin | §14 row 3 |
| coalescing the 820 copies/token into fewer | **~0**: 347 MB in 27.1 ms is 12.8 GB/s, 95% of the 13.4 GB/s the PCIe probe measures | §13 |
| fewer kernels (M38) | 0 wall clock | M38 |

M28 §6's "`plan` 0.85 → 0.4 ms/layer ⇒ TPOT −40%" is withdrawn for the second time (M28 §7(b) suspected it, §8
measured it, this section re-measures it on the current binary): the device is *busy* during the routing read, so the
host round trip is not 0.85 ms of device idle per layer. What remains is the 2.3-3.9 ms of genuinely empty window.

**The shape of the gap to Strata**: naiwa occupies the device 43.7 ms of a 50.8 ms step — 17.1 arithmetic, 27.1 its
own PCIe staging, 9.4 waiting for the CPU, 7.1 idle. Strata's window is 34.7 ms of stream with 15.4 arithmetic,
11.3 `waitB` and 7.7 `waitCPU`, and **no comparable staging stage at all**. naiwa pays ~21 ms of *exposed* copy time
that Strata does not pay; that, not the routing round trip, is the 10-11 ms.

## 15. `--pcie-frac` sweep: the current default 0.4 is on the CPU-bound side, 0.6 is the knee

Requested experiment ("reduce the PCIe share and measure"). It turned out to have the opposite sign from the
assumption: **less PCIe is worse, and the default is not the optimum.**

Protocol: one engine process per arm, 3 consecutive 128-token greedy requests, no diagnostic env (no
`NAIWA_STEP_TIME`, no `ROUND_STATS`), no MTP, 2.5 GiB cache, lookahead default, `--pcie-frac` the only difference.
Two passes in opposite orders (`0.4→0.2→0.1`, then `0.1→0.2→0.4`, then `0.6→0.8→1.0`, then `0.7→0.6→0.5→0.4`).
Unit = median of the 3 requests, ms/token.

| `--pcie-frac` | pass A | pass B | pass C | pass D | best estimate |
|---:|---:|---:|---:|---:|---:|
| 0.1 | 53.7 | 53.6 | | | 53.6 |
| 0.2 | 53.1 | 52.9 | | | 53.0 |
| **0.4 (today's default)** | 48.2 | 48.3 | | 47.7 | **48.0** |
| 0.5 | | | | 47.4 | 47.4 |
| **0.6** | | | **46.4** | **46.4** | **46.4** |
| 0.7 | | | | 47.8 | 47.8 |
| 0.8 | | | 49.7 | | 49.7 |
| 1.0 | | | 52.6 | | 52.6 |

Within-arm spread across the 3 requests is 0.2-0.9 ms and the same arm reproduced to 0.1 ms across two passes in
opposite orders, so the 1.4-1.6 ms between 0.4 and 0.6 is real and the curve is a genuine interior optimum:

```
 53.6 ──0.1        46.4 ──0.6  <- knee
 53.0 ──0.2        47.4 ──0.5
 48.0 ──0.4        47.8 ──0.7
                   49.7 ──0.8      52.6 ──1.0
```

### 15.1 Why: the two sides are not symmetric, and the CPU side is the expensive one

Instrumented pair (`NAIWA_STEP_BUDGET=1 NAIWA_ROUND_STATS=1`, medians over 128 steps, same 3-request protocol):

| | frac 0.4 | frac 0.6 | Δ |
|---|---:|---:|---:|
| wall ms/token | 48.6 | 47.5 | −1.1 |
| `total` (sum of phases) | 28.60 | 27.45 | −1.15 |
| `expert_dev` (GPU expert kernels) | 14.00 | **18.25** | **+4.25** |
| `plan` | 21.45 | 21.25 | −0.2 |
| `staging_wait` | 2.50 | 2.20 | −0.3 |
| GPU-resident hit % of selections | 63.4 | 65.0 | +1.6 |
| staged over PCIe % | 18.9 | **24.8** | **+5.9** |
| CPU experts % | 17.8 | **10.2** | **−7.6** |
| staged MiB / step (steady state) | ~123 | ~170 | +47 |

Turning the knob moves ~7 points of selection (34 experts/step) from the CPU to the GPU. That **costs 4.25 ms of GPU
expert kernel time** and **saves 5.4 ms** — i.e. a CPU expert costs ~0.15 ms of *exposed* step time (12.9 ms CPU gemv
/ 85 CPU experts = 0.15, and §14 shows the CPU is waited for, not overlapped), while the same expert on the GPU costs
~0.12 ms and overlaps with the copy engine. So the marginal expert belongs on the GPU **until PCIe saturates**, which
is what the right-hand side of the curve is: at 0.8-1.0 the staged share is 33-40%, the copy chain grows to ~26-30 ms
and becomes the binding stage again (§14: copies are the binding chain at ~27 ms).

This is the same trade §14 named from the other end: the step is `max(kernel chain, copy chain)` and 0.6 is where the
two chains are closest to balanced.

### 15.2 Consequences

1. **`--pcie-frac 0.6` is worth ~1.4-1.6 ms/token (3%)** over the shipped default, with no change to bytes, kernels,
   cache size or drafting. It is a one-line default change, but it should not be made the default on this evidence
   alone: the curve is sharp (0.5 → 47.4, 0.7 → 47.8) and the optimum is machine- and prompt-dependent, so it needs
   the second prompt and the 154-turn chat before it becomes a default.
2. **The oracle is unaffected**: it runs `frac 0`, where staging is disabled, so this knob cannot move it. What it
   does change is which tokens are produced at `frac > 0` (placement changes rounding order, §10/M39), so this is a
   performance default, not a correctness one.
3. **Reducing the PCIe share was the wrong direction**, and the reason is now measured rather than assumed: the CPU
   experts are not overlapped, so every point of share moved to the CPU adds ~0.15 ms per expert to the critical path
   while it removes only ~0.1 ms of copy time.

## 16. Step-time distribution, naiwa at 0.4 and 0.6, against Strata's latest numbers

Same protocol for both naiwa arms: one engine process, 3 consecutive 128-token greedy requests, `NAIWA_STEP_TIME=1`
(the only diagnostic), no MTP, 2.5 GiB cache. Steps are taken from `STEP_TIME seq_begin/step_end` pairs, prompt
steps and the first decode step dropped (they carry prefill and first-touch allocation: 0.5-1.2 s). 381 steps per arm.

| ms/step | frac 0.4 | frac 0.6 |
|---|---:|---:|
| n | 381 | 381 |
| min | 36.6 | **36.0** |
| p10 | 42.5 | **39.9** |
| p25 | 45.0 | 42.2 |
| **p50** | **47.7** | **45.5** |
| p75 | 51.7 | 48.8 |
| p90 | 57.1 | **52.4** |
| p99 | 67.1 | 61.1 |
| max | 69.6 | 69.0 |
| mean | 48.9 | 45.9 |
| sd | 5.9 | 5.2 |
| share ≥50 ms | **35.4%** | **18.9%** |

Histogram (% of steps):

| range | 0.4 | 0.6 |
|---|---:|---:|
| <40 | 3 | 11 |
| 40-42 | 5 | 11 |
| 42-44 | 10 | 17 |
| 44-46 | 17 | 17 |
| 46-48 | 18 | 14 |
| 48-50 | 12 | 10 |
| 50-52 | 12 | 8 |
| 52-55 | 10 | 5 |
| 55-60 | 8 | 4 |
| ≥60 | 6 | 2 |

So `--pcie-frac 0.6` is not a shift of the median: it **halves the upper tail** (≥50 ms: 35.4 → 18.9%) and lifts the
floor (<40 ms: 3 → 11%). Mean improves 3.0 ms while the median improves 2.2 — the tail is where the win is.

### 16.1 What the two engines' distributions can and cannot be compared on

Strata exposes **no per-window timing** (`STRATA_DECODE_TIMING` prints one line per request over 128 windows;
`STRATA_VERIFY_PROFILE` prints one aggregate stage profile; there is no per-window dump in `src/`, re-checked with
`grep getenv`). Its request-level samples from the latest runs (`/tmp/decode-strata-timing.log`, 8 requests, 2 engine
sessions):

| | ms/window | note |
|---|---:|---|
| min | 36.97 | warm, session 1 |
| p50 | 38.12 | |
| max | 39.75 | session 2 |
| mean | 38.16 | |
| sd | 1.02 | |

Because each of those is a **mean of 128 windows**, its sd cannot be compared with a per-step sd: with 128 samples,
`se = sd_step/√128 = 1.02` implies a per-step sd of up to ~11.5 ms. So Strata could be as spiky as naiwa's 5.2 ms
and this data would look identical. **The step-level dispersion comparison is not available; what is available is the
composition.** (Both engines are stable at request granularity: naiwa 0.6 gives 46.28 / 45.92 / 45.97, Strata
37.0 / 37.2 — so naiwa's variance is *intra*-request, i.e. step to step, not request to request.)

### 16.2 Composition at 0.6 (nsys) against Strata's stream

naiwa arm at `--pcie-frac 0.6`, same union/gap analysis as §14 (span 49.1 ms/token under nsys, ~3 ms of profiler
overhead against the clean 46.0):

| per token | frac 0.4 | **frac 0.6** | Δ | Strata |
|---|---:|---:|---:|---:|
| span / window | 50.8 | **49.1** | −1.7 | 37.4 |
| occupancy (union) | 43.7 | 43.0 | −0.7 | 34.65 (stream) |
| arithmetic kernels | 17.1 | 17.4 | +0.3 | **15.4** |
| copies (`memcpy`) | 27.1 | **31.6** | **+4.5** | not reported; `waitB` 11.3 is the wait on them |
| **wait for CPU experts** (`doorbell_wait`) | **9.4** | **3.0** | **−6.4** | `waitCPU` 7.7 |
| device idle | 7.1 | 6.1 | −1.0 | not reported |
| host `plan` | 21.45 | 21.25 | −0.2 | 0.06 |

That is the mechanism in one table: the knob converts **6.4 ms of CPU-expert waiting into 4.5 ms of PCIe copying**,
and the copies overlap better than the wait, so the net is −1.7. After the change naiwa's CPU wait (3.0) is *below*
Strata's (7.7); what remains is copies (31.6 vs their 11.3 of `waitB`) plus 6.1 ms of idle, against essentially
matching arithmetic (17.4 vs 15.4).

### 16.3 Where the remaining 7.3 ms sits (45.5 vs 38.2)

Median step 45.5 vs Strata's mean window 38.2 = **+7.3 ms (19%)**, down from the +10-11 ms of §11. Decomposed by
§16.2 and §14, in the order they are worth attacking:

1. **PCIe copy exposure** — 31.6 ms of copy-engine time for ~170 MiB/step, of which ~24 ms is exposed (only ~7 ms
   hides behind kernels). Strata's equivalent wait is 11.3. This is now the single largest block.
2. **6.1 ms of idle**, 58% of which follows a D2H read (§14.1) — the device-side plan's ceiling.
3. **Arithmetic 17.4 vs 15.4** — 2 ms, and the only part M38 already showed cannot be won by launch count.

And the floor says something useful: naiwa's best decile (p10 39.9, min 36.0) is already **at** Strata's average
(38.2). The gap is not a constant offset; it is the upper half of our own distribution.

Evidence: `/tmp/pure-pure-naiwa3.log` (0.4), `/tmp/ps-f06.log` (0.6), `/tmp/ns-f06.csv`,
`/tmp/decode-strata-timing.log`.
