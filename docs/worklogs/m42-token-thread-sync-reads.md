# M42 — the token thread's synchronous reads

**Status:** 🚧 in progress (2026-11-11). **Parent:** M41 §「Why Strata's kernel staging wins and naiwa's loses」.

**Current milestone summary:** [近期优化、五轮性能锚点与剩余验收](m42-optimization-summary.md). The sections below preserve chronological findings and corrections; the summary does not promote experimental switches to defaults.

M41 ended with a measured contradiction: with `--pcie-frac` aligned to Strata's staged share (naiwa 14.2%, Strata 14.6%) naiwa still runs 45.9 ms/token against Strata's 37.2, while doing **less** CPU work (25.7% of selections against 38.8%) and holding **more** experts resident (60.1% against 46.6%). The work partition is therefore not the explanation, and neither is the transport mechanism (M41 measured both fetch variants slower and explained why).

This milestone attacks what the timeline says instead: the device goes idle right after the token thread performs a **synchronous read**, because that read empties the launch queue and the host needs time to refill it.

## 1. Where the idle is (measured, current binary)

Capture: `nsys` at `frac 0.6`, spin layout (5 workers + participating caller, 20 ms), no MTP, one 128-token request. Decode suffix derived from the request's own duration; every union and gap is computed on the merged GPU timeline across both streams.

Layer structure, keyed on `doorbell_copy_f32_kernel` (the top-k publish, one per layer):

| per layer | p50 | p90 | p99 | mean | mean × 48 |
|---|---:|---:|---:|---:|---:|
| publish→publish window | 885.5 µs | 1441.7 | 5961.1 | 1055.6 | 50.67 ms |
| publish end → first resident expert kernel | 128.7 µs | 290.9 | 858.9 | 165.5 | **7.94 ms** |
| copy-stream work inside that window | 103.2 | 284.2 | 796.3 | 142.2 | 6.83 ms |
| **uncovered part of it** | **26.4** | 45.2 | 60.1 | **23.2** | **1.12 ms** |
| idle in the whole window | 70.3 | 207.9 | 2694.6 | 167.7 | 8.05 ms |

Two conclusions, one of them against the plan I proposed at the end of M41:

1. **The globally uncovered portion is ~1.1 ms/step; this is NOT a device-dispatch speedup ceiling.** The copy stream is busy during much of this window, but earlier compute could still overlap those copies and shorten a dependency chain. The previous claim that dispatch can save at most 1.1 ms is withdrawn.
2. **The idle is elsewhere.** Attributing every merged-timeline gap ≥10 µs to the pair of operations on either side gives 5.40 ms/step of attributable idle in total:

| ms/step | occurrences/step | gap between |
|---:|---:|---|
| **3.14** | 33.2 | `[CUDA memcpy Device-to-Host]` → `[CUDA memcpy Host-to-Device]` |
| 0.43 | 0.9 | `copy2d_f32` → `[CUDA memcpy Host-to-Device]` |
| 0.41 | 4.4 | `[CUDA memcpy Host-to-Device]` → itself |
| 0.24 | 14.0 | `usilu_bf16` → cutlass GEMM |
| 0.15 | 10.7 | `q4_hc_combine` → `gemv_kernel_batched` |
| 0.15 | 10.2 | `badd_bf16` → `q4_hc_combine` |

The dominant bucket is one per layer that has CPU misses: a 30 KB **D2H** (the CPU experts' activation) after which nothing runs for ~100 µs until the host issues the next **H2D**. That is the token thread draining the queue in `CpuMiss::submit`'s `x2d.to_device(&Device::Cpu)` — a blocking `clone_dtoh`, which is a stream-wide `synchronize()`.

M40 measured this boundary as *host* time (10 µs/layer, `spawn` phase) and correctly called it cheap for the host. It is not cheap for the **device**: unblocking the host is followed by building and submitting the CPU job before any GPU work is queued again, and the device waits through it. 3.14 ms/step is the largest single attributable idle class, and it is a mechanism, not a noise estimate.

## 2. The mechanism

Replace the blocking activation read with an **asynchronous, mapped handoff**, reusing the primitive the lookahead already uses for exactly this tensor:

- at the top of the layer (where `Preload::submit` already publishes `x2d` for the lookahead), also publish it into a channel the CPU expert job can read;
- the publish is a kernel on candle's stream, so it is ordered after the kernels that produce `x2d` and costs the host one queue call, no sync;
- `CpuMiss::submit` passes the channel and sequence to the worker; the worker waits for the sequence and reads the mapped bytes, instead of the token thread blocking on a D2H.

Nothing else changes: same activation values, same expert ids and weights, same fused kernel, same result path. The measurable prediction is specific: the `D2H → H2D` idle class shrinks and clean TPOT falls, with the CPU job's inputs bit-identical.

## 3. Gates

| gate | requirement |
|---|---|
| activation identity | the bytes the worker reads equal `x2d.to_dtype(F32)` for the same layer (unit test on a synthetic tensor, bitwise) |
| oracle | `frac 0` + `--no-lookahead`, `head -1` byte-identical to `harness/m1/reference.oracle.txt` |
| determinism | three runs of one greedy request at the fixed `frac` give the same text (`sha256`) |
| unit tests | `cargo test --release --features cuda --bin naiwa-rs` |
| mechanism KPI | nsys: `D2H → H2D` idle class ms/step, and total idle; plus `spawn`/`plan` phases |
| performance | diagnostic-free alternating arms, ≥3 rounds, 128 tokens, explicit `--mtp-n-predict`; medians only |
| cleanup | if the idle class shrinks but TPOT does not, record it and delete the mechanism |

## 4. Steps

- [x] **S1 — locate the idle.** §1. Device-side dispatch demoted to ~1.1 ms/step; the blocking read promoted to the top item.
- [ ] **S2 — the handoff.** Second activation channel sized like the lookahead's, published at the layer head, read by the CPU worker after a sequence wait. `NAIWA_CPU_ACT_ASYNC=1`, default off.
- [ ] **S3 — identity test** for the published bytes, then the oracle and the determinism gate.
- [ ] **S4 — measure** the idle classes and the clean arm; keep or delete on evidence.

## S5 — the expert split, half to the GPU and half to the CPU

Strata's intra-layer batching turned out to be **token** grouping (`G = split_ && T >= 2`, `tb_/te_` token ranges, `post(l,grp)` then `pre(l+1,grp)`), so with our T=1 it degenerates. The user's actual proposal — split each layer's **experts**, one half on the GPU and one on the CPU, overlapped — is exactly our `--pcie-frac 0` shape (only residents on the GPU, every miss on the CPU), which had only ever been measured under the old Rayon/no-participation structure. Re-measured clean, current binary, spin layout, async activation, no MTP, 3 requests/arm, two rounds in opposite order:

| frac | GPU share | CPU share | round 1 | round 2 | TTFT |
|---:|---|---:|---:|---:|---:|
| 0.0 | residents only | ~44.7% | 52.06 | 52.47 | ~1060 ms |
| 0.1 | | ~35% | 47.76 | 47.88 | 750 |
| 0.2 | | ~26% | 46.24 | 46.26 | 580 |
| 0.6 (reference) | | ~9.5% | 46.07 | 45.96 | 500 |

Half-and-half costs **+6 ms/token** and +560 ms of TTFT. Two facts fall out of this table:

1. **The split barely matters between 0.2 and 0.6** (46.24 against 46.07). Over that range the staged byte volume changes ~2x and the CPU share ~2.7x, and the step time does not move. Neither the PCIe chain nor the CPU chain is binding there, so the ~46 ms is dominated by the per-layer structure rather than by where the experts run. This is exploratory evidence of a flat region, not proof that neither resource binds: offsetting CPU/PCIe costs can produce a flat total. There were only two restart rounds and the 0.6 reference did not fix all experimental variables.
2. **The CPU cannot absorb half.** `NAIWA_RET_DBG` at frac 0 (43.2% share): queue wait p50 11.7 us, per-job compute p50 642.6 us, mean 782.4 us. With ~48 jobs/token that is **~37.5 ms of CPU wall per token**, against a 52 ms step. Strata runs 38.8% of the same selections in 21.2 ms/window, i.e. **113 us per expert-row against our ~150-180 us**: its CPU row is 1.4-1.6x cheaper. That, plus its non-blocking per-layer handoff, is why it can afford the split and we cannot.

**Correction:** the claimed ROUND_STATS instrument defect was an analysis error. `rounds` is cumulative, whereas CPU milliseconds cover the reset window (normally four passes). Dividing by cumulative rounds caused the spurious undercount. In `/tmp/step-half-ret0.log`, excluding the first prefill/startup window, dividing each CPU total by four gives mean 30.33 ms/pass, median 30.15. These are accumulated CPU job wall times, not GPU-exposed waits. Also withdraw the 1.4–1.6x CPU-kernel comparison above: Strata host-window time and naiwa job compute do not share a measurement boundary.

The frac sweep changes workload allocation and is not a fixed-partition scheduling test. It does not reject expert-level overlap, nor establish a CPU-kernel prerequisite. Next isolate submission ordering with partition, operators, and bytes held fixed by implementation.

Evidence: /tmp/m42-half.out, /tmp/pf-half-{0.0,0.1,0.2}-r{1,2}-*.json, /tmp/step-half-ret0.log, /tmp/half-ret0.out.

## S6 — fixed-policy resident-first submission (withdrawn)

**Status:** closed, no retainable benefit. See `harness/m42/resident-order-experiment.md`, `resident-order-ab.sh`, and `resident-order-results.json` for implementation outline, parameters, and raw captures (full SHA256).

Changed only resident submission position in source: plan → resident launch/fence → staging → synchronous CPU activation/job → staged → unchanged sum. Async activation disabled; frac=.25; same spin/layout/operator settings. No drafting; CLI rejects explicit `--mtp-n-predict` without `--mtp`, so it was omitted with this documented exception.

| restart round | baseline median | resident-first median | delta ms/token |
|---|---:|---:|---:|
| 1 | 44.977 | 46.967 | +1.990 |
| 2 (reverse) | 45.688 | 47.763 | +2.076 |
| 3 | 45.631 | 46.899 | +1.269 |

Each cell is three 128-completion-token requests; denominator 127. No performance failures/retries. Build passes; 58 tests passed, 5 ignored; frac0/no-lookahead oracle first line byte-identical. But frac=.25 SHA256 differs within both arms, so actual per-token demand/bytes were NOT proven equal. Do not describe this as strict same-work proof. This implementation was reverted. Plausible source mechanism: synchronous activation D2H now waits behind resident kernels and delays CPU start; requires event timeline to establish causality.

Next: establish synchronous-path per-token repeatability (locate first route/cache divergence) before claiming fixed-work scheduling A/B. Separately fix mandatory CPU activation channel lifecycle, not merely ring depth; predictions may be dropped, CPU inputs may not.

## S7 — identify the first repeatability boundary

**Status:** completed diagnostic control, not TPOT A/B. Full evidence and source boundary: `harness/attribution/results/repeatability.md` and `.json`. Same baseline in six restarted processes per condition, frac .25, no drafting, async activation off, only lookahead toggled between conditions.

Lookahead on: all three pairs first differ in residency and final GPU/CPU assignment while IDs/weights at that record still match; first indices 5/5/98. Later weight divergences 6/6/99 and ID divergences 10/10/99. Two pairs still return identical text. Lookahead off: ALL six complete 6192-record traces and text SHA256 match. This identifies lookahead-dependent residency timing as the first observed divergence boundary in this configuration, not an individual copy-corruption bug. Do not disable lookahead by default based on diagnostic runs.

Probe fixes: independent per-process work.jsonl avoids logger JSON corruption; residency sampled under the exact classification mutex, not a later reacquisition. First attempt had corrupted records and was not treated as complete evidence. Tests: Python 8; Rust 58 passed/5 ignored.

Next use no-lookahead as a same-condition diagnostic control, or instrument prefetch commit timing to make the production lookahead path attributable. Matching work records still do not prove activation identity or transferred-byte identity.

## S8 — ordered execution checklist and CPU-before-DMA

Checklist: `harness/attribution/TODO.md`. T1–T4 complete for repeatable no-lookahead control; T4b second prompt/production verification next. T5 mandatory async activation and T6 lookahead event attribution remain separate.

Default-off `NAIWA_CPU_BEFORE_DMA=1` submits complementary CPU experts immediately after unchanged staged-set selection, before staging DMA. No workload or arithmetic reordering intended. Tests/oracle pass. Three diagnostic pairs of 18480 records (three requests/process) match exactly. Clean AB/BA/AB medians 43.177→42.270 / 43.508→42.289 / 43.439→42.603 ms/token; paired median improvement .907 ms/token. SSE p95/p99 improve each round, not engine-step percentiles. Corresponding request-position text hashes all match: global hash inequality reflects first-vs-reuse request differences, not arm mismatch.

Retained default off pending second prompt and lookahead-on verification. See `harness/attribution/results/cpu-first.md` and archived diagnostic/clean JSON for evidence and limits. CPU critical-path causal explanation still requires event correlation, not inferred from cumulative timings.

## S9 — second prompt and lookahead-on follow-up

T4b completed. Detailed tables, clean manifests, diagnostic work hashes and limits: `harness/attribution/results/cpu-first-followup.md`.

- Second prompt (OS thread scheduling), no-lookahead: clean paired TPOT deltas -1.580/-.991/-1.258 ms/token, median -1.258. SSE p95/p99 improve each round. Text matches at corresponding request positions. Separate diagnostic: ALL three pairs of 18336 work records match exactly.
- Database prompt, lookahead-on, otherwise same frac .25/spin/no-MTP setup: deltas -.114/-.601/-.620 ms/token, median -.601. Text diverges at every request position across restarts; only policy-level end-to-end result, not identical-work attribution. SSE p95 round1 regresses slightly, p99 improves all three.

Retain default-off CPU-before-DMA. This is not default production verification (default Rayon/frac .4, MTP/chat still outstanding). T4c next: correlated event boundary measurement, then default-config/MTP/chat adoption gates. Do not infer exact CPU critical-path reduction solely from aggregate TPOT.

## S10 — default Rayon/frac .4 and MTP2 adoption checks; timeline confound

See `harness/attribution/results/cpu-first-default-and-timeline.md` and archived manifests/tails.

Clean Rayon/frac .4/lookahead-on: no-drafting deltas -.574/-1.126/-1.021 ms/token (three wins). Fixed greedy MTP2: -1.003/-.891/+1.352 (two wins, one material regression). Text divergent at corresponding request positions, policy-level results only. No default adoption; MTP step acceptance and multi-turn checks remain.

Separate three-pair no-lookahead CPU_TIMELINE+WORK_TRACE capture matches all 6192 work records/pair, but startup timing is confounded: MOE_WORK formatting/write lies before baseline CPU submit and after early CPU submit. Therefore ~300us/layer observed startup advance is NOT valid quantitative production mechanism evidence. Clean performance captures do not enable that trace. Need buffered timing events, explicit request/step identity and GPU-clock correlation. T4c-b stays open rather than declaring causal closure.

## S11 — buffered host/CPU schedule evidence

T4c-b1 complete, b2 GPU/result/request correlation still open. See `harness/attribution/results/cpu-first-buffered.md` and `.json`. Three AB/BA/AB diagnostic rounds without WORK_TRACE/CPU_TIMELINE: CPU submit ~26us/layer earlier, worker24–32us earlier, compute return18–40us earlier; staging enqueue11–13us later. Worker arithmetic interval not consistently faster. No dropped events;6192 layer records and5654 CPU jobs/process. Tick>=3 filter is local, not guaranteed decode-only. Do not turn differences into a TPOT sum.

New default-off NAIWA_SCHEDULE_FILE buffers numeric timestamps; serialize after inactivity. Python analyzer validates artifacts/phase completeness. Initial compile failure accidentally ran stale binary (/tmp/attr-cpufirst-buffered-01); invalid capture excluded, missing-artifact rejection added. Valid capture -02. Rust58 passed/5 ignored; Python9 passed. CPU-first still default off pending MTP/chat and correlated GPU evidence.

## S12 — CPU result publication is cheap in the repeatable control

Optional executor job tags add buffered publication begin/return timestamps. Three AB/BA/AB no-lookahead/no-MTP captures, filtered5558 jobs/arm/round: mean1.4–1.5us/p99~2.2–2.4us, max<57us, no>=100us. No large host publication/backpressure stall observed in this condition. GPU consumption remains unmeasured; do not generalize to MTP or multiply into TPOT benefit. Evidence `harness/attribution/results/cpu-first-publication.md` and `cpu-first-publish.json`, raw /tmp/attr-cpufirst-publish-01.

Previous MTP r3 last logger window differs (base67.1%/len2.34 vs early61.0%/len2.22), but5s windows are not request-aligned totals. Clue only, not causal explanation. Per-step acceptance still required. Rust58/5ignored, Python9 pass; defaults unchanged.

## S13 — MTP verification-count confound measured

Buffered per-sequence verify outcomes, three diagnostic AB/BA/AB rounds, default Rayon/frac .4/lookahead-on/MTP2. Full evidence `harness/attribution/results/cpu-first-mtp-attribution.md`, counts and exact decomposition JSON alongside.

New r2/request2 regresses39.40→41.26ms/token even though amortized client decode ms/outcome decreases90.98→90.34: outcomes55→58, accepted/proposed64.5%→58.6%. Exact symmetric count/cost decomposition +1.858 = +2.142 count -.284 cost ms/token. All9 paired requests lower amortized cost; verify count changes -4..+3 and can reverse TPOT. This is accounting attribution, NOT pure verify-kernel time or causal same-work standardization. Original clean r3 lacks counts; do not retrospectively claim explained. Diagnostic timing separate from clean evidence.

MTP acceptance root still open (lookahead residency/CPU-GPU arithmetic/drafter nondeterminism). Next negative controls no-lookahead+drafter determinism; GPU consume correlation/chat still incomplete. Defaults unchanged. release check/build and Rust58/5ignored, Python10 pass.

## S14 — harness MTP report and no-lookahead control

Reusable `analyze.py mtp` validates and decomposes per-request counts/cost; runner now validates full MTP/schedule artifacts. Python11 tests pass. New negative control: MTP2/no-lookahead ALL9 paired requests match entire ordered [proposed,accepted,keep_len] trace and text. Separate clean AB/BA/AB deltas -.900/-1.021/-.561ms/token, median-.900, corresponding-position text equal. Evidence `harness/attribution/results/mtp-nolook-control.md` and JSON.

This supports lookahead-dependent execution variation as observed acceptance-divergence trigger in this configuration, not universal drafter determinism or proof about historical clean regression. CPU-first alone does not produce observed acceptance divergence here. Defaults unchanged; production prefetch event correlation/GPU consumption/chat still open.

## S15 — refreshed same-session Strata comparison

Clean three-arm rotated ABC/BCA/CAB, no drafting,128tokens x3requests/process x3restarts. Current nai​​wa default Rayon/frac .4 vs CPU-first vs Strata existing config. Process-median medians48.485 /47.792 /37.136ms/token. CPU-first gap difference-of-medians10.657ms (+28.7%); median paired gap10.640. Internal paired CPU-first deltas -1.077/-.528/-.530 (median improvement.530). All9 Strata requests accepted0of0 drafts. See `harness/attribution/results/strata-three.md` and JSON/log extracts for settings/limits. Historical spin/frac .6 comparison not directly comparable; no regression claim. CPU-first default remains off.

## S16 — large-cost triage

See `harness/attribution/results/large-cost-triage.md` plus manifest/GPU ledger. Clean current/disable-promotion/disable-lookahead3-arm: disabling promotion costs paired median1.246ms/token, disabling lookahead .550; do not remove them merely to cut bytes. These are policy ablations, not isolated copy costs.

Current Rayon/frac .4/CPU-first nsys: last5952 layer windows normalized to124 48-layer-window equivalents (NOT independently verified decode steps), span50.602, idle7.456, copy union27.367/exclusive17.628, result-wait union7.616/exclusive4.613, other kernels17.903ms. H2D347.261MB/equivalent. Large exposure categories identified; NOT additive cross-engine gap attribution or recoverable-time bounds. Need same-request byte-source accounting and GPU/request clock boundaries. Defaults unchanged.

## S17 — Strata and naiwa payload ledgers

Opt-in Strata per-layer selected staged bytes/counts/position/CPU callback buffered to existing request profile boundary; naiwa staging/cache-fill byte events. Three diagnostic restart pairs raw /tmp/attr-engine-ledger-03. Strata decode selected payload155.604/141.586/143.551MB per request position, repeated exactly; naiwa tick>=3 stage190–192MB plus cache-fill169–170MB per48-layer equivalent. Windows not aligned; no causal subtraction or bandwidth speedup claim. Large candidate: extra fills ~47% payload, but clean disable-lookahead regresses, so need duplicate/useful-byte identities and demand queue priority, not delete policy. Evidence `harness/attribution/results/engine-ledger.md/json`, Strata patch/provenance archived. Failed -01/-02 rejected. CUDA builds, Rust58/5ignored, Python11 pass; HIP/SYCL and patched-Strata clean identity gate not done. GPU/request bridge remains open.

## S18 — fill lifecycle and budget narrowed

Opt-in numeric expert/slot enqueue+publish+promotion+resident-use events;6diagnostic processes show36.5–38.0% cache-fill bytes superseded before any classified resident hit, tails censored separately. No claim GPU use or request alignment. Existing --lookahead-swaps2/1/0 clean rotated3restarts:1 improves .837/.464/1.007ms/token vs2;0 wins vs2 but loses to1 all3rounds. Separate diagnostics2->1 cache-fill169->91MB, demand staging190->207MB, net~60MB/equivalent less; unused superseded fill payload~64->32MB.0 transfers least but slower than1, so overlap/placement not merely bytes. Evidence `harness/attribution/results/fill-budget.md/json`, configs/manifests. Default unchanged; second prompt/MTP/chat and GPU consumption gate open. CUDA build/Rust58+5ignored/Python12 pass.

## S19 — budget1 follow-up across prompts, drafting and original submit order

Clean3restart AB/BA/AB comparisons budget2->1: second prompt no drafting improves .377/.770/.437ms/token; original MTP2 improves3.080/1.972/.762; second MTP2 improves3.029/.959/5.332; CPU-first OFF original no drafting improves1.029/.981/.680. All policy comparisons/text differs, not same-work. Independent original MTP diagnostic all9 D/N lower (89.50–92.18 ->84.71–86.97ms), outcome counts -4..+2. Cannot backfill counts to clean runs or treat D/N as pure verify. See `harness/attribution/results/fill-budget-validation.md/json` and archived manifests/decomposition. Budget1 remains opt-in; chat/tails/TTFT/correctness adoption gates and GPU/request correlation open. No engine changes this follow-up.

## S20 — same-session candidate vs Strata refresh

Clean rotated3arm3restarts: base default Rayon/frac .4/budget2/CPU-first off, candidate CPU-first+budget1, original Strata zero drafting(all9confirmed). Medians49.013/47.635/37.903ms/token, candidate +25.676% vsStrata; median paired gap9.757ms, combined internal improvement.968ms. Evidence `harness/attribution/results/strata-budget1.md` and manifest/report/draft checks. No cross-session gap subtraction or MTP comparison. Defaults unchanged.

## S21 — Strata per-step GPU partition

Export existing raw globaltimer profile per window with pos/T, host reach-wait per layer and tail sync. Three diagnostic restart pairs,384T1windows/process validated; exact disjoint stream partition closure. Mean9requests GPU span35.345ms: dense/router11.194, shared.884, plan.283, resident1.538, staging-wait-fetch11.354, staged.522, CPU wait7.965, result/tail.386, gaps.034, head1.185. Host reach12.003/pool22.116/tail1.207 overlap GPU, not additive. Naiwa worker summed per ALL layer windows24.54–25.26ms vs Strata callback22.12 is NOT same boundary/work. Matching naiwa GPU partition still missing; cannot numerically allocate10ms gap yet. See `harness/attribution/results/engine-step-costs.md/json`, step_costs.py and refreshed Strata patch. Python14tests/CUDA Strata build pass, HIP/SYCL untested. Current capture nai​​wa budget2, not latest candidate1; diagnostic not clean.

## S22 — both-engine diagnostic partitions and overhead control

Added default-off NAIWA_GPU_PARTITION_FILE9 CUDA events/layer +next route. V1 hot allocation/JSON/file added5.92ms, excluded from clean attribution. V2 event reuse/background numeric buffer median+.969ms, still perturbs. Three cross-engine restarts at CPU-first/budget1,378structural full48layer groups/process: non-MoE/progression19.627vs13.283, plan/preload3.653vs.277, resident4.001vs1.539, staging10.488vs11.352, staged1.560vs.521, CPU result8.898vs8.277, tail.712vs.382. Separately closed GPU ledgers48.938vs35.631, difference13.307 NOT clean9.732; endpoints/outer costs/probe differ, residual not hidden. Largest candidate now non-MoE/framework progression+host plan, not merely bytes. Full semantics and remaining causal gaps `harness/attribution/results/structural-attribution.md`, JSON aligned/round/raw manifests. Python15/Rust58+5ignored pass; CPU-only CUDA lookup regression found/fixed. Exact causal clean-gap allocation still open; no default changes.

## S23 — model interior partition and shared ordering test

Fork optional qwen4_split callback exports layer/mixer/MoE/routed/shared boundaries; reusable CUDA events, no candle operator changes.3restart probe/control delta median+.500ms. Naiwa mixer7.15–7.20 vs separate Strata6.38–6.40ms: not main6ms unresolved bucket alone. Naiwa HC/setup5.47–5.55, shared1.94, token/head gap3.17–3.69; boundaries differ/router still in routed. Simple shared-before-routed experiment clean original deltas-.595/-.084/-.181; second+.125/-.484/-.673. Marginal, reverted behavior; probes retained default-off. Evidence `harness/attribution/results/model-attribution.md/json`, saved patch and manifests/reports. Release build/check and Rust58/5ignored/Python15 pass. Defaults unchanged; no complete causal clean10ms allocation claimed.

## S24 — HC narrow: mix vs norm/inject

Expanded default-off model boundaries12events/layer,3restart capture /tmp/attr-hc-partition-02. Mean attn mix2.197 +mlp mix2.109=4.306ms within HC/setup; norm/setup.740, combine/norm.364, inject gaps.236/.234. Existing Strata firstHC stamps norm.332/down.811/up-post.808ms. Naiwa firstmix2.197vs Strata down+up1.619 descriptive only (~.58), not equal operator/context; secondhalf lacks split, don't double. Naiwa already uses fused hc_norm/hc_mix/hc_combine; next targets low-rank down/up/intermediate path, not generic missing fusion. Failed-01 idle-flush validator race rejected; runner waits2s before model/GPU validation, new-02 all3rounds passes. Evidence results/hc-narrowing.md +hc-model-costs.json/overhead. Release build/check and Python15 pass; defaults unchanged.

## S25 — HC projections and sparse-probe control

Default-off down/act/up markers18events/layer. Dense probe adds1.55–3.03ms, not usable clean attribution. New stride8 sparse adds.006/.235/.236ms, matched selected GDN layers8–40 down18–21us/up20–24us/act7–9us/reduce4–5us; separate Strata firsthalf down16–17us/up-post17us. Modest per-layer candidate, no48layer extrapolation (allGDN sample). Layer0 anomaly includes startup/boundary uncertainty. Secondhalf Strata31->17 includes router, not up-only. See results/hc-projection-narrowing.md/json. No operator/default change; release check/Rust58+5ignored/Python15 pass. Next priority host route/plan and head/token boundaries, not broad HC fusion.

## S26 — tail/head narrow

Default-off tail-only4reused CUDAevents/token, new next_rows filter excludes prompt transitions. /tmp/attr-head-02 three restart pairs377samples/process: finalHC.044ms, LMhead1.113–1.117, posthead->nextmodel1.883–1.899. Initial-head-01 without next_rows includes crossrequest gaps, not used. Probe delta-.389/+.340/+.833ms, median+.340; don't subtract constant. Strata head~1.19 separate capture, LMhead not main difference. Stable1.89ms posthead candidate contains sampler/state/scheduler/setup, not pure sample or recoverable benefit. Active sampler branch unmeasured, do not assume CPU fallback. Evidence results/head-narrowing.md/json. Release check/Rust58+5ignored/Python15 pass; no defaults changed.

## S27 — sampler active path

Default-off host sampler entry/exit and branch reasons via boundary callback. Three independent3restart diagnostic sets; last /tmp/attr-sample-input-01 eachprocess384calls,378CPUinput/5devicegreedy/1temperatureSome,0logprobs/speculative/multi/processors. Initial host sampler~1.26ms/call includes possible synchronization/filtering, not pure argmax. Pipeline has try_sample_batch_cuda then into_cpu_rows fallback; exact caller/eligibility rejection not yet correlated, don't assert bug/bypass. Need pre/post fallback and request/step IDs. Evidence results/sampler-narrowing.md +sample-host-costs.json/sample-input-branches.json. Release build/check/Rust58+5ignored/Python15pass; defaults unchanged.

## S28 — actual sampler fallback and opt-in fast path

Found try_sample_batch_cuda rejects single-seq noncategorical after plan checks, then copies logitsCPU. New NAIWA_SINGLE_GPU_SAMPLE=1 permits ONLY existing argmax plan/no sampling_logprob_required through existing top1batch/row, all eligibility guards retained; defaultoff. Clean original gains1.743/.907/1.811ms, second1.681/1.887/2.221, no-lookahead2.824/2.403/1.969 with full corresponding-position text equal across6processes. Independent diagnostics384reject+Sampler calls baseline->0optin; head1.11unchanged, posthead1.88–1.91->.54–.57ms. Concrete~1.3ms diagnostic tail reduction, clean gain not additive oldStrata gap. Evidence results/single-gpu-sampling.md +manifests/reports/patch. Oracle/cornercases/MTP/chat not yet passed, keepdefaultoff. Release check/Rust58+5ignored/Python15pass.

## S29 — GPU sampling verification follow-up

Secondprompt no-lookahead3restart gains2.702/2.088/1.957ms with ALLtext identical. New serial3turn harness appends actual replies/saves bodies; chat TPOT improves2.439/1.042/1.421 and SSEp99 all improve, BUT TTFT+22.6/+183.3/+345.7ms, different histories so fixedtranscript control needed. Logprobs capture extended;9pairs nonempty JSONlogprobs+text exact byposition, but paired medianTPOT+.447ms, fallback nonregression not passed. Evidence results/validation-followup.md/json/manifests; no defaults/engine changes. Python15tests pass after mockmessages compatibility fix. Oracle/cornercases/chat corpus/fallback performance gates open.

## S30 — pool tuning on optimized combination

Current CPUfirst/budget1/GPU sampling with Rayon vs spin8 vs physical5(5worker+caller,20ms spin,affinityCPU1–5+0).3restarts rotated each mode. MTP2 medians39.442/50.103/36.739ms; no drafting45.973/56.145/42.537. Physical5 paired gains2.704/1.712/3.854MTP and2.951/3.661/4.062no draft. Joint count/affinity/dispatcher test, not isolatedspin. Spin8 regression~10ms. Initial no-draft-01 outertimeout excluded/ownedPG cleaned; valid new-02 complete. No same-sessionStrata comparison, acceptance/workidentity unproven. Evidence results/pool-tuning.md, manifests/reports. Defaults unchanged; secondprompt/chat/diagnostics open. Python15pass.

## S31 — tuned pool same-session Strata refresh

Clean3rotated restarts bestphysical5/MTP2 vsRayon/MTP2 vsStrata no drafting. Medians37.956/38.910/37.795ms; best-Strata deltas+.161/-1.356/+.410, median+.161(+.426%). Approxparity, notstablewin, asymmetric MTP. PoolvsRayon-1.345/-2.380/+.598, so earlier allroundgain not fully replicated. No acceptance counts; no speculative cause claim. Evidence results/strata-best-pool.md, report/manifest/draftchecks; all9Strata0of0. Defaults unchanged.

## S32 — five restart suffix drafting comparison

User requested5rounds; new complete5round three-arm rotated run /tmp/attr-strata-suffix5-01 (45requests). Medians nai​​wa best/MTP2 36.805, Strata suffix0 37.841, suffix3 37.794ms. Candidate paired improvement vsSuffix3 .935/.006/1.094/1.535/-.426,median.935, wins4of5notstable; candidate variance higher. Suffix actually10accepted/15proposed over15requests (first0/0, reused1/2+1/1 eachrestart), tinycoverage; notneuralMTP. Evidence results/strata-suffix5.md+manifest/report/logs. Defaults unchanged;5roundorder notfullybalanced(6needed).
