# M41 — CPU submission before staging, then device resident dispatch

**Status:** in progress. **Parent:** M40 (reopened; its closure is unsupported), M39 routing timeline.

## Scope and acceptance

First isolate one mechanism: select the staged experts exactly as before, submit the remaining CPU work before ring acquisition and DMA preparation. Keep the resident/staged/CPU partition, weights and ordering unchanged. `NAIWA_EARLY_CPU=1` enables the experiment; unset retains submission after staging. Ring creation and selection still precede submission. No candle changes.

Then evaluate device-side resident dispatch as a separate mechanism. It must handle mixed-hit layers, maintain residency-map lifetime/fences, and include a stale-map negative control. This second phase is not implemented yet.

| Gate | Requirement |
|---|---|
| Build/tests | release CUDA build and existing binary tests |
| Partition | unchanged selected IDs, CPU IDs, weights and ordering on identical routing inputs |
| Oracle | frac 0, no lookahead, first line byte identical |
| Performance | diagnostic-free alternating arms, at least three rounds, 128 generated tokens; record full text hashes |
| Decision | keep only a demonstrated benefit; no inference from phase timing alone |

## M40 corrections (supersede its closure)

RET_DBG reports averages over 48 completed jobs, not per-job percentiles. Its compute span includes conversion, publication and any diagnostic output. Stage+spawn medians from another run do not establish a per-layer start-delay distribution. The old prediction probe has no request/step identity and mismatched accounting windows; its overlap and residency ratios are not decision evidence. Already resident predictions are not copied bytes. Predicting expert IDs from a different activation cannot produce reusable expert outputs. CPU result wait and routing/dispatch idle are separate overlapping mechanisms, not a shared 3 ms ceiling.

## Implementation

- CPU early submission callback runs after the existing `streamed_misses` selection and before ring acquisition/copies.
- It uses a filtered copy of misses, preserving row order and weights. The staging path still builds GPU hits in its original order.
- Missing staging facilities fall back to the original CPU submission path.
- Default remains unchanged. Phase budgets move CPU submission into Stage when enabled and must not be compared as though their boundaries were unchanged.

## Evidence / remaining work

Implementation and release build started. Benchmark, partition checks, oracle, and keep/revert decision pending. No performance claim yet. Device resident dispatch remains the next separately measured phase.

## Experiment result — CPU submission before DMA

Release CUDA build passed; binary tests: 56 passed, 4 ignored. Same-session order B1/E1/B2/E2/B3/E3, one fresh engine per arm, three consecutive 128-token requests per engine. Explicit `--mtp --mtp-n-predict 2`, frac 0.6, cache 2.5 GiB. No timing/prediction probes enabled. Numbers below are medians of the three request TPOTs (ms/token).

| round | baseline | early CPU | difference |
|---|---:|---:|---:|
| 1 | 42.154 | 42.817 | +0.663 |
| 2 | 42.148 | 42.979 | +0.831 |
| 3 | 41.945 | 43.871 | +1.926 |

No observed benefit. Full output hashes differ both within arms and between arms; MTP workload is therefore not controlled. This does not prove a causal regression or close the CPU overlap direction. Partition equivalence was not runtime-verified, nor were task-level boundaries correlated. The experiment also allocated a filtered CPU list in both arms and ran early submission under the staging-ring lock; these limitations matter for interpretation.

Decision: remove the experimental callback, switch and reordered submission. Original scheduling restored; do not ship an unproven mechanism. The experiment's oracle was not run; no oracle-pass claim is made for it. Device resident dispatch remains unimplemented and the milestone stays open. Next required work is a controlled routing/task identity trace before another scheduling experiment, then a separate device dispatch implementation.

Evidence: `/tmp/m41-ab.out`, `/tmp/pf-m41-{base,early}-{1,2,3}-{1,2,3}.json`, `/tmp/m41-build.log`, `/tmp/m41-tests.log`. Restored build: `/tmp/m41-restored-build.log`.

## Follow-up: per-task host boundaries

Added default-off `NAIWA_CPU_TIMELINE`. Identity is process-local `(layer, layer tick)` with row count, padded expert IDs and exact f32 weight bits. It does not yet carry engine request IDs. One isolated 128-token request, explicit MTP depth 2, frac 0.6. Excluding layer ticks <=2 leaves 2478 jobs (2435 three-row jobs, 43 one-row jobs). No duration-based outlier filtering.

| interval (microseconds) | p50 | p90 | p99 | max |
|---|---:|---:|---:|---:|
| Route::submit return to CpuMiss entry | 49 | 89 | 157 | 246 |
| activation D2H call | 16 | 17 | 21 | 83 |
| before executor submission to closure entry | 11 | 15 | 23 | 106 |
| closure entry to fused computation return | 776.5 | 1442 | 2123 | 4825 |

The submission interval includes channel preparation and queue insertion, not just queue residence. Compute includes CPU activation conversion; fallback jobs would require a separate end marker. All retained jobs finished computation after the host queued their result upload (median difference 680.5 us). This is NOT GPU waiting: actual device execution can occur later. Logging before publication perturbs scheduling, so the request TPOT is not performance evidence.

Finding: ordinary task-start overhead is much smaller than computation, while task computation has a substantial tail. This motivates correlating GPU wait kernels with task completion and row/expert work before another scheduling change. It does not prove CPU calculation is exposed, nor set an improvement ceiling. No resident-dispatch implementation yet.

Release CUDA build passed; diagnostic request completed. The earlier 56-test result predates this probe. Evidence: `/tmp/step-m41-timeline.log`, `/tmp/m41-timeline-build.log`, `/tmp/m41-task-analysis.py`. Probe disabled by default; no scheduling or arithmetic change.

## Workload stratification of the same diagnostic capture

Three-row CPU tasks by nonzero expert-row entries (padding excluded):

| active pairs | jobs | p50 us | p90 us | max us |
|---|---:|---:|---:|---:|
| 1 | 435 | 355 | 451 | 3176 |
| 2 | 651 | 585 | 703 | 1429 |
| 3 | 344 | 804 | 938 | 2209 |
| 4 | 420 | 1000 | 1183 | 2590 |
| 5 | 207 | 1195 | 1455 | 2485 |
| 6 | 212 | 1364 | 1627 | 4825 |
| 7 | 75 | 1644 | 1860 | 2262 |
| 8 | 59 | 1753 | 2069 | 2162 |
| 9 | 23 | 1969 | 2223 | 2299 |

Much of the aggregate tail tracks workload: typical latency rises from 355 us for one pair to 1969 us for nine. 48/2478 jobs exceed or equal 2 ms, totaling 108.224 ms of CPU closure duration (not GPU exposure or recoverable time).

Residual examples after stratifying by layer, rows, active-pair count and distinct-expert count, using groups with >=5 jobs: layer 40/tick25 (3 rows, 1 pair) 3176 us versus group median 367.5; layer0/tick25 (3 rows,6 pairs,6 experts) 4825 versus 1440; layer39/tick25 (3 rows,3 pairs) 2209 versus 757. Worker-start delays for these three are 17/10/11 us respectively: their large duration occurs after closure entry, not in the outer executor queue.

These groups do not control exact expert IDs, activation values, cache/page residency or CPU contention. There are no exact expert-row assignment groups with >=5 repeats in this capture. The tick25 cluster suggests checking common step-level conditions, but does not identify scheduling as the cause.

Source inspection: fused_cpu.rs sorts/deduplicates `(expert,row)` pairs and runs gate/up then down reduction with an internal parallel dispatcher. Thus the outer executor's 11 us start delay does not measure internal worker dispatch/wake/barrier costs. Next discrimination should measure those two compute phases plus scheduler/page-fault evidence and GPU result wait. Prediction overlap is not a substitute for this measurement.

## Phase attribution follow-up

Separate diagnostic request, frac 0.6, explicit `--mtp --mtp-n-predict 2`, with CPU_TIMELINE and FUSED_PHASE_DBG. Matched each phase line to the following CPU_TASK on the serial executor; 2494 retained jobs after ticks <=2 excluded, no unmatched CPU_TASK lines. This is a new workload, not a speed comparison with the prior request.

| phase us | p50 | p90 | p99 | max |
|---|---:|---:|---:|---:|
| activation quantization | 30 | 46 | 55 | 81 |
| gate/up phase excluding activation quantization | 476 | 902 | 1373 | 4776 |
| intermediate quantization | 9 | 19 | 28 | 60 |
| down phase excluding intermediate quantization | 256.5 | 494 | 735 | 1289 |
| CPU closure through fused return | 859 | 1497 | 2182 | 5035 |

Largest spikes: layer0/tick14, eight pairs, gate IQ3_XXS/down IQ4NL: gate/up 4074 us, down 790 us, total 5035 us. Layer30/tick14, only two pairs, gate IQ3_XXS/down Q2_0: gate/up 4776 us, down 131 us, total 5032 us. The latter is a gate/up-localized anomaly, not merely a larger task. Timing does not separate internal parallel dispatch/wake/barrier, weight access and arithmetic.

Source search finds no install_task_dispatcher call in naiwa src or mistralrs-core src; the normal fused path therefore appears to use its Rayon pool (default capped at eight workers). The comment about a custom spin pool is not evidence that one is installed. Confirm actual worker threads before proposing a pool change.

Median closure-minus-phase residual is 81 us, including phase debug output, conversion, accounting and unmeasured preparation. PHASE_DBG us_pairs is not a valid measure of pair construction: its timer starts after pair construction and data access. Do not use it to claim zero preparation overhead. Logging perturbs publication and scheduling; no TPOT claim.

Evidence: /tmp/step-m41-phases.log, /tmp/m41-phases.out. Next boundary to measure is inside gate/up parallel execution, separating dispatcher arrival, worker work and completion/barrier, alongside scheduler/page-fault evidence and GPU exposure.

## Dispatcher boundary probe

Default-off NAIWA_DISPATCH_TRACE wraps each internal task with monotonic timestamps and atomic aggregates. It measures first task start, overall parallel call span, last task-body end to caller return, max task wall time, and summed task wall time. Sum is not occupancy or CPU time. Atomics and timestamps perturb scheduling; last-end timestamp precedes bookkeeping, so tail includes instrumentation completion.

One separate 128-token diagnostic request, explicit MTP2, frac0.6. After ticks <=2 excluded: 2491 jobs, exactly two dispatch records per CPU job, zero mismatches.

| phase / metric (us) | p50 | p90 | p99 | max |
|---|---:|---:|---:|---:|
| gate/up first start | 23 | 29 | 40 | 780 |
| gate/up dispatch span | 464 | 897 | 1351 | 3237 |
| gate/up last task end to return | 24 | 47 | 294 | 2592 |
| gate/up longest task | 145 | 187 | 519 | 2450 |
| down first start | 13 | 23 | 38 | 3133 |
| down dispatch span | 267 | 492 | 772 | 6828 |
| down last task end to return | 17 | 33 | 126 | 3562 |
| down longest task | 38 | 76 | 143 | 1757 |

Layer8/tick56: gate/up 20 tasks, span3237 us, first25, tail2592, max task158. Its down120 tasks: span6828, first3133, tail3562, max task28. Layer9/tick56 gate/up tail2394 of3006; layer10/tick56 tail1811 of2481. This cluster localizes a major class of stalls outside the task bodies, consistent with completion synchronization/caller rescheduling rather than slow dot arithmetic. It does not distinguish Rayon completion machinery from OS descheduling or probe effects. Another class exists: layer40/tick3 gate/up span2610, max task2450, tail46, so not every spike has the same cause.

The prior gate/up-only diagnosis was capture-specific; this capture also has severe down dispatch stalls. Next causal check should collect caller/worker scheduler state and off-CPU durations, then test a dispatcher change only if these waits are exposed on the GPU critical path. Do not convert these diagnostic maxima into TPOT savings.

Evidence: /tmp/step-m41-dispatch.log, /tmp/m41-dispatch.out, /tmp/m41-dispatch-build.log, /tmp/m41-dispatch-check.log. Release build and diagnostic request completed. No numerical or scheduling policy change; probe defaults off.

## Scheduling follow-up

Prior dispatch capture: gate/up spans sum1307.780 ms, first-start58.382 and tail95.973; down spans736.961 ms, first43.290 and tail63.593. Combined first/tail261.238 ms across59 retained layer ticks. Intervals >=500 us contribute37.159 ms (7 first-start intervals,26 tails). These are diagnostic host intervals, not TPOT savings. Most boundary cost is ordinary small intervals, not the largest spikes.

perf software events are denied (perf_event_paranoid=4); no system setting changed. A separate diagnostic request used proc task schedstat/stat sampling nominally every50 ms. Confirmed eight cpu-moe workers plus cpu-experts. Decode span5.629434 s; counters below cover only first-to-last in-window samples, not full request.

| thread | runtime ms | runnable wait ms | major faults |
|---|---:|---:|---:|
| cpu-experts |755.5|28.9|0|
| cpu-moe-0 |1943.7|57.0|0|
| cpu-moe-1 |1862.3|111.0|0|
| cpu-moe-2 |1854.7|97.7|0|
| cpu-moe-3 |1844.9|81.1|0|
| cpu-moe-4 |1839.5|54.8|0|
| cpu-moe-5 |1798.5|139.3|0|
| cpu-moe-6 |1739.1|195.3|0|
| cpu-moe-7 |1645.2|281.6|0|

There is runnable scheduling delay and uneven worker service, but no task-level causal association yet. Sleeping/futex time is not included in runqueue wait. No major faults were observed on these threads during sampled decode, which does not exclude cache misses or bandwidth contention. Summed worker times overlap and cannot be converted to TPOT. Sampling and diagnostic logging perturb scheduling.

Evidence: /tmp/m41-sched-samples.json, /tmp/step-m41-sched.log, /tmp/m41-sched.out. Next useful controlled experiment is worker-count tuning (existing NAIWA_CPU_MOE_THREADS), with diagnostic-free alternating trials and output/MTP work accounting, before implementing a new dispatcher.

## Worker-count experiment: 8 versus 6

Diagnostic-free alternating order 8/6/8/6/8/6, fresh engine each arm, three consecutive128-token requests each, explicit MTP2, frac0.6, cache2.5 GiB. NAIWA_CPU_MOE_THREADS is the only experimental setting. No busy-poll dispatcher was implemented or tested in this experiment.

| round | 8 workers median ms/token | 6 workers median | delta6-8 |
|---|---:|---:|---:|
|1|42.872|43.840|+0.968|
|2|43.663|44.632|+0.969|
|3|44.968|44.943|-0.025|

Pooled nine-request medians43.088 versus44.574. Output hashes differ within/between arms and periodic MTP acceptance reports differ; workload is not controlled. Baseline round medians drift upward, and the ordering is not counterbalanced. Therefore this is evidence against adopting six workers on this workload, not a causal measurement of the scheduling cost or proof of eight being optimal. Keep eight as the busy-poll experiment baseline.

Evidence: /tmp/m41-threads.out, /tmp/pf-m41-t{6,8}-r{1,2,3}.log and corresponding request JSON captures. No code/default changes. Bounded busy polling remains untested; require a separate implementation and correctness gate before claiming any benefit.

## Rayon caller wait-path inspection

Rayon-core1.13.0 registry::in_worker_cold injects a StackJob and calls LockLatch::wait_and_reset for a caller outside the pool. LockLatch uses Mutex<bool> plus Condvar; it has no bounded-spin stage. The separate worker sleep path yields for32 rounds before its sleep transition. Worker idle polling and external caller completion waiting are distinct mechanisms.

A separate request with explicit MTP2 and frac0.6 sampled cpu-experts wchan/stat during decode (5.394046s). Of249 in-window samples:231 futex_do_wait/S,11 zero/S,4 zero/R,3 futex_do_wait/R. Reads are not atomic; mixed states can reflect races. Proc scans ran nominally every10ms but actual sampling was slower. Sleeping includes useful waiting for parallel work and executor idle parking, not merely delayed completion notifications. No stack trace or job-level association was available, so these samples do not locate all waits at LockLatch.

This narrows a future busy-poll experiment to caller completion waiting as well as worker idle policy. A worker-only spin pool experiment would not isolate the caller wakeup hypothesis. Do not interpret231/249 as CPU waste or an improvement ceiling. Evidence: /tmp/m41-wchan-samples.json and /tmp/step-m41-wchan.log. No code changes in this follow-up.

## Isolated dispatcher benchmark

Implemented and ran standalone CPU-only comparison in [harness README](../../harness/m41/README.md). Real Rayon versus explicitly labelled Strata-inspired scheduling model, identical synthetic task bodies, six compute participants, three rounds with reversed second-round ordering. All final outputs matched serial reference. Hot18-task p50 medians19.26/10.13us (Rayon/model with participating caller); requested500us gap57.82/28.93us.120-task hot73.36/61.73us, gap158.31/113.32us. Work differs between task-count cases; only within-case mode comparisons are valid. Not real Strata ExpertPool numbers or engine gains. No engine scheduling changes.

## Actual Strata native CPU pool benchmark

Standalone harness now calls real ExpertPool::run_split_multi_native on actual GGUF layer0 weights. See harness/m41/README.md and results/native-pool.csv. Host participation versus waiting, equal six compute participants,18 tasks/phase, no affinity, fixed inputs and IDs. Three rounds: two-expert median-of-medians199.886 vs203.482us; six-expert598.935 vs605.206us. Two-expert paired result reverses in round2; six-expert differences vary1.984-37.119us. No evidence for a large stable benefit from host participation alone. Full native Rayon-versus-Strata arithmetic-aligned comparison remains unmeasured. Within-process output checks passed; cross-arm checksums match but full cross-arm bitwise comparison not yet performed.

## Matched native arithmetic and task partition

Implemented actual Rayon callback dispatch of Strata native row kernels, matching ExpertPool's18 global row-range tasks per phase. Real GGUF layer0, same activation/IDs, six compute participants, affinity off, no intentional gap. Every output byte matches an actual ExpertPool reference. See harness/m41/README.md and results/native-rayon.csv. Three rounds with reversed second-round order. Two-expert p50 median-of-medians198.883us ExpertPool versus283.262us Rayon; six-expert611.439 versus704.152us. Median paired differences82.926us and31.078us respectively. This establishes a dispatch-path difference without changing arithmetic, not a pure busy-poll effect or engine benefit. It still combines caller participation, task allocation, completion waits, instrumentation/bookkeeping differences and Rayon FFI callback overhead. No default changes.

## Three-row gap experiment and actual-pool spin control

Extended matched native benchmark to3 rows/expert and requested500us inter-batch sleep. All outputs byte-identical to real ExpertPool reference. Three rounds: median paired Rayon overhead34.304/55.684us for2/6 experts without intentional gaps,400.393/587.340us with500us gaps. Full-overlap routing and hot repeated weights, not sparse engine replay.

Actual Strata counter-control, fixed2 experts/3 rows/500us gap, only STRATA_POOL_SPIN_US changed:0us p50s524.725,416.181,524.956;2000us268.935,271.009,276.960. Median paired improvement247.996us. Source default worker spin is20ms. Thus worker idle policy is causally relevant in the isolated test; generic statements that only caller completion matters are insufficient. No attribution solely to futex latency: frequency/cache/scheduling effects remain possible. No engine default changed. Evidence and schema in harness/m41/README.md, results/native-rayon-rows3.csv and native-spin-control.csv.

## Adoption decision after spin budget scan

Actual ExpertPool budgets0/50/200/1000/2000us, three rounds: p50 medians524.485/523.474/382.128/272.522/271.700us. Short50us ineffective,200us unstable,1ms retains2ms latency. Whole-process CPU seconds0.3779/0.3990/0.4366/0.5739/0.5508; includes setup/reference/warmup, so not an isolated cost measurement. Full output comparisons passed. Evidence results/spin-budget.csv and harness README.

Decision: do not introduce as naiwa default on current evidence. Approve only a separately gated opt-in engine experiment if pursued: worker count/task partition held fixed, explicit spin budgets0 and1000us, request completion correctness/oracle and fixed-input equivalence, alternating >=3 rounds128 tokens with explicit MTP depth, record actual speculative workload and CPU cost alongside TPOT. Goal remains stable1ms/token end-to-end; microbenchmark deltas cannot satisfy it. No engine dispatcher implementation or end-to-end busy-poll test exists yet.

## Engine physical-core layout correction

User clarified one compute thread per physical core, aligned with Strata. The earlier eight-worker pinning on five physical cores was the wrong comparison and is not evidence against physical-core affinity. Corrected opt-in NAIWA_CPU_PHYSICAL_LAYOUT: five spin workers pin to logical CPUs1..5; dispatch caller pins toCPU0 and participates in task claiming. CPU0/6,1/7,... are SMT pairs on this host. Other engine threads remain unrestricted: this does not guarantee exclusive core ownership or match all Strata thread placement. Affinity and host participation happen inside the experimental dispatcher; NAIWA_CPU_MOE_THREADS must be5. Spin budget1000us, original task partition unchanged. This is a combined layout/caller/dispatcher comparison, not isolated affinity.

Three fresh-engine arms per configuration, three consecutive128-token requests per arm, explicit MTP2, frac0.6, cache2.5GiB. Order base/physical,physical/base,base/physical. No diagnostic probes.

|round|Rayon8 median ms/token|physical5+caller median|delta|
|---|---:|---:|---:|
|1|42.900|43.487|+0.587|
|2|43.161|43.296|+0.135|
|3|42.728|42.483|-0.245|

No stable speed gain. Full output hashes vary within and between arms; MTP work differs. Tail-latency benefit has not been measured. No default adoption justified. Actual weight/dispatcher equivalence and oracle for the new engine implementation remain unverified; standalone exactly-once and panic tests predate the physical-layout extension. Source build passed. Keep experimental path default-off pending stronger checks or remove if this direction is abandoned. This is not the same as Strata's default20ms spin or18-task phase partition.

Prior unpinned engine rounds, Rayon42.003/41.378/42.781, spin1ms44.447/44.050/43.957, spin forever45.642/46.541/43.372 ms/token: no speed gain observed. Wrong-layout eight-worker pinning yielded about60-64ms/token and is superseded by corrected comparison above. CPU utilization is not a rejection criterion; only speed, tail behavior and correctness matter.

Evidence: /tmp/m41-physical.out and /tmp/pf-physical-{base,five}-r{1,2,3}-*.json; /tmp/m41-physical-build.log.

## Implementation overhead correction and spin-budget follow-up

Found repeated caller sched_setaffinity on every dispatch and clock reads on every worker spin iteration. Caller pin now cached per thread; budget clock checked every1024 polls. These fixes were not individually performance-isolated. Physical-layout hardcoded CPU0..5 remains machine-specific; unrelated engine threads are not isolated.

Added isolated fused_quantized_output_matches_rayon_bits test: Q4_0 gate/up/down,3 rows, padding and shared experts. Reference computed before custom-dispatch installation, then100 custom calls compared by every f32 bit. Passed in default experimental pool and separately physical5+caller with20ms budget. This covers callback correctness on this fixture, not all model quant formats or engine oracle. Exactly-once repeated-batch and panic/reuse tests passed. Release build passed.

Same corrected binary and physical5+caller layout:1ms versus20ms worker spin, three fresh-engine rounds,3 consecutive128-token requests/arm, explicit MTP2,frac0.6,cache2.5GiB. Order1/20,20/1,1/20. No diagnostics.

|round|1ms TPOT median|20ms median|delta20-1|
|---|---:|---:|---:|
|1|43.175|42.288|-0.887|
|2|41.966|41.260|-0.706|
|3|41.768|41.984|+0.216|

No stable large gain established;20ms wins two rounds and loses one. Text hashes vary, so actual speculative workload differs. No current-binary Rayon comparison or step-tail measurement was performed in this batch, and old sessions cannot establish a speedup. Keep default Rayon. The revised pool remains opt-in and oracle pending. Evidence /tmp/m41-fixed-budget.out, /tmp/pf-fixed-spin-{1000,20000}-r{1,2,3}-*.json, /tmp/m41-spin-equivalence.log, /tmp/m41-spin-fixed-build.log.

## Engine tail latency comparison

Current corrected binary, Rayon8 versus physical5 workers+participating caller on cores0..5 with20ms worker spin. Three fresh engines/arm,3 consecutive128-token requests each, explicit MTP2,frac0.6,cache2.5GiB. Order R/S,S/R,R/S. STEP_TIME/ALL enabled for both; no per-task probes. Client content-event timestamps captured. Diagnostic timing TPOT is not a clean speed A/B. Prompt-marked spans excluded, no duration filtering. Nearest-rank p95/p99.

|round|Rayon step n|spin step n|Rayon p95/p99 ms|spin p95/p99 ms|
|---|---:|---:|---|---|
|1|179|182|111.163 /116.257|108.432 /116.229|
|2|181|177|111.497 /124.832|108.506 /113.539|
|3|175|175|112.842 /122.908|108.191 /122.099|

Pooled decode step distribution: Rayon535 steps, p50=92.229,p95=111.768,p99=122.281,max132.092ms; spin534 steps,p50=89.9475,p95=108.465,p99=116.229,max122.146ms. Step p95 decreases each round; p99 also numerically decreases each round, first round effectively unchanged. MTP step count/output hashes differ; this is observed distribution, not fixed-work causal attribution.

Nonempty SSE-content event gaps after first content:1143 intervals/arm. Pooled Rayon p95=106.268,p99=114.954,max132.283ms;spin p95=104.164,p99=111.598,max122.185ms. Per-round SSE p99 Rayon115.249/120.104/114.954 vs spin111.734/110.395/115.913: third round slightly worse. No >=150ms interval in either arm. Events are not individual tokens; burst emissions give near-zero median gaps. TTFT excluded.

By request ordinal, SSE p99 first/second/third: Rayon121.502/113.006/114.617 vs spin111.734/113.243/111.598ms. Second request effectively tied. Small sample, no confidence interval or request-level bootstrap, no extreme-tail claim beyond observed values.

Conclusion: consistent step-p95 improvement and modest observed pooled p99 improvement, but no stable SSE-p99 improvement in all rounds. This supports retaining opt-in for further validation, not changing default. CPU usage is not a gate. Oracle still pending. Raw summary harness/m41/results/engine-tail-summary.json; logs /tmp/pf-tail-{rayon,spin}-r{1,2,3}.log and JSON captures; script /tmp/m41-tailbench.py.

## Requested spin-on versus Strata comparison

Spin enabled explicitly for benchmark launches, not made global default: physical5+caller,20ms, frac0.6,cache2.5GiB,explicit MTP2, existing promotion settings. Three fresh-engine rounds each, three128-token requests, order N/S,S/N,N/S. No timing probes. Naiwa round TPOT medians41.845/42.483/42.212ms. Strata37.033/37.075/37.127ms; median round gap5.137ms,13.9% relative to Strata. Output work differs; not same-token causal comparison. Source parameters are best observed candidates, not a global optimum.

Critical limitation: actual Strata args have --spec4 but NO --mtp runtime directory. Logs show0/0 drafts on first requests and only suffix drafts on later requests. Thus Strata is NOT MTP-enabled and must never be presented as both engines with MTP. Existing available mtp-component.gguf has blk.48.* Q4_0 tensors; Strata loader expects dense.txt/dense.bin/experts.bin. Existing mtp_rt converter expects mtp.layers.0.* tensors with Q2_0 fused experts. No compatible runtime directory found under home/tmp search. True both-MTP comparison remains blocked by artifact preparation, not completed. Initial script generation failed before valid measurements (heredoc collision), then corrected.

Raw request captures /tmp/pf-final-{naiwa,strata}-r{1,2,3}-*.json; summary harness/m41/results/strata-comparison.json. Current experiment shows naiwa spin+MTP2 still behind Strata without its MTP, not evidence that spin closes the gap. No global defaults changed.

## Both engines with actual drafting disabled

Three rounds N/S,S/N,N/S, fresh engine per arm, three128-token requests each, no diagnostic timing. Naiwa keeps physical5+caller20ms spin,frac0.6,2.5GiB cache; --mtp and --mtp-n-predict removed. Strata no --mtp, --suffix-draft0 and --lookup-chain0; --spec4 retained solely because native IQ path rejects spec<2. All nine Strata requests report drafts accepted0 of0. Neither engine is proposing drafts. Other settings unchanged from preceding comparison.

Round median TPOTs Naiwa: 45.904, 46.683, 45.819; Strata: 37.047, 37.041, 37.071 ms/token. Median of round medians 45.904 versus 37.047, gap 8.857ms (23.9% relative to Strata). Denominator is reported completion_tokens minus1 (127). Request medians mix first and two warm requests; separate ordinal metrics and full text hashes retained in results/no-drafting-comparison.json. Output equivalence is not assumed. No global default changes.

Evidence /tmp/m41-nospec.out, /tmp/m41-nospec-strata-engine.log, /tmp/pf-nospec-{naiwa,strata}-r{1,2,3}-*.json.

## Current spin/no-drafting attribution refresh

Load logs: naiwa1317 resident slots within2.5GiB requested; Strata1028 slots1.98GiB. Hence naiwa is not smaller in resident capacity on this run. Slot counts differ in per-expert bytes and placement; they are not an equal-budget experiment.

Current naiwa diagnostic (physical5+caller20ms,frac0.6,no MTP): ROUND_STATS32 four-pass windows averaged resident66.706%,staged23.806%,CPU9.488%; staged229.582MiB/pass; CPU job wall10.753ms/pass. Includes first-pass effects; not exact same output as Strata or nsys capture. Strata timing refresh first request CPU4.13/resident4.27/PCIe1.60 experts per layer-window, later3.88/4.66/1.46. Strata CPU wall22.33/20.97ms/window yet request37-39ms: cannot compare CPU wall as exposure. Different division of work, more CPU share at Strata, despite smaller resident cache.

New naiwa nsys capture under same configuration: request reported TTFT501,total6954ms. Approximate suffix window uses last6453ms of GPU events, divided by127 actual decode steps. This is request-duration-derived, not event-aligned request boundary; final event alignment and first/last-step edges are uncertain. Profiler span50.811ms/step is NOT clean45.9ms TPOT. Union42.162, idle8.649, arithmetic-only union17.456, memcpy/memset union31.456, CPU-result spin union1.842ms/step. Exposed spin = union(all)-union(without spin)=0.776; exposed copies=22.864ms/step. Components overlap and do not sum. Host-to-device400.568MB/step (~899 ops), D2H0.623MB, D2D66.824MB. Copy bytes field rounded in CSV; approximate. No comparison to Strata waitB as though identical metrics.

Interpretation: current CPU result wait alone cannot explain the9ms clean gap. Copy exposure and routing/dispatch idle are the primary candidates left by this capture. No numeric allocation of the gap without matched boundaries. CPU-share differences explain why making naiwa CPU scheduler faster need not close the gap. H2D is larger than staged accounting (229.6MiB) but those are separate runs, so the residual cannot simply be declared speculative-prefetch waste. Next experiment should identify H2D producers and correlate actual loads with consumption, or disable/reduce lookahead as a single-mechanism control.

Evidence /tmp/step-attribution-current.log,/tmp/attribution-current.out,/tmp/m41-attribution-strata-engine.log,/tmp/nsys-m41-current.nsys-rep,/tmp/m41-current-trace.csv. No code/default changes in this refresh.

## Lookahead causal ablation, current spin/no-MTP configuration

Fixed physical5+caller20ms,frac0.6,cache2.5GiB,no MTP; only --no-lookahead toggled. Three fresh-engine rounds, three128-token requests/arm, order on/off,off/on,on/off; no diagnostic probes. Round median TPOT on46.495/45.992/45.970 vs off46.960/47.042/47.025ms. Disabling costs0.465/1.050/1.055ms. Text differs; workload equivalence not asserted. Evidence favors retaining lookahead on this prompt, not deleting speculative traffic wholesale.

Separate off nsys capture compared with previous on capture, approximate suffixes derived from request elapsed decode times, normalized by127 actual steps. On/off H2D400.572/319.510MB, copy union31.456/25.334ms, arithmetic17.456/16.701, spin1.842/1.098, exclusive spin0.776/1.098, exclusive copy22.864/23.731, idle8.649/8.659. Fewer copied bytes but higher exclusive-copy interval; overlap pattern and residency changed. CPU-wait kernel duration does NOT increase materially here, so prior predicted large CPU-wait rebound is unsupported. Exclusive activity is a descriptive overlap statistic, not a causal critical-path saving. Profiles are single captures and window boundaries approximate. Their spans50.811/50.189ms contradict clean A/B direction; do not claim a profiler wall-clock speed result or explain full clean delta from these captures.

Source inspection verify.cpp post path supports DMA/kernel/direct PCIe modes; default option string auto follows kernel-mode selection in shown setup path. Actual mode should be confirmed from runtime before equating Strata waitB to transfer activity. Native grouped kernels, flags and fetch_blobs differ from naiwa's async memcpy staging; no comparable Strata byte count yet.

Conclusion: wholesale lookahead removal does not recover the9ms gap. Preserve lookahead and focus on actual required-transfer bytes, placement/consumption, and post-routing dispatch idle. H2D reduction itself is not proof of optimization. Raw summary harness/m41/results/lookahead-attribution.json; /tmp/m41-lookahead.out; /tmp/nsys-m41-current-nolook.nsys-rep; /tmp/m41-nolook-trace.csv. No code/default changes.

## Matched-text Strata PCIe transport ablation

Actual Strata, drafting disabled (no MTP, suffix0,lookup0,spec4 native compatibility), cache800 setting. Only explicit --pcie-mode kernel versus dma. Three fresh-engine rounds, three consecutive128-token requests each; mode order K/D,D/K,K/D, no timing probes. All nine corresponding full text SHA256 values match across modes, with the same ordinal sequence e19ca091/daaedd64/c651bc5c in each round. This controls output text, not proof of identical internal routing/cache history.

|round|kernel TPOT median|DMA TPOT median|DMA-kernel|
|---|---:|---:|---:|
|1|37.153|39.431|2.277|
|2|37.189|40.868|3.679|
|3|37.229|39.231|2.002|

Paired median difference2.277ms/token. Stable direction, variable magnitude; DMA request times also vary strongly. Native grouped arithmetic unchanged. Source: kernel mode publishes mapped expert aliases; GPU fetch_blobs loads full native blobs into device staging then rebase_ptrs before grouped compute. DMA mode supplies device staging pointers and queues copies through fetch callback. Thus kernel mode is not zero-transfer: full expert bytes still cross PCIe as GPU loads. Transport initiation, synchronization, overlap and device resource usage differ. Busy flags/waitB cannot be treated as isolated memcpy time.

This establishes transport path as a material performance factor inside Strata, not a2.277ms guaranteed naiwa saving or an additive portion of the8.9ms cross-engine gap. Naiwa stages three projections with per-slot fences and host submissions; replacing that with a descriptor-driven GPU fetch is a distinct implementation requiring memory-registration lifetime, slot-fence correctness and numerical/output gates. Next useful test is the same selected experts/bytes under DMA versus GPU fetch in naiwa, preserving routing/admission and ordering.

Evidence /tmp/m41-pcie.out,/tmp/m41-pcie-strata-engine.log,/tmp/pf-pcie-{kernel,dma}-r{1,2,3}-*.json; persisted summary harness/m41/results/strata-pcie-mode.json. No code/default changes in this ablation.

## naiwa GPU fetch staging: built, measured, reverted

Minimal slice of the Strata kernel-staging idea: naiwa_fetch_projection (vectorised16-byte path, bytewise fallback) reading the already-registered host expert bytes into the existing staging slot on the existing copy stream, between the slot read fence and the layer completion event. Selected experts, bytes, slot and ordering unchanged; only the transfer mechanism differs. Gated by NAIWA_STAGE_GPU_FETCH.

Byte equality test passed for1,4097,65539 bytes with source offsets0,0,3 (registered host source, device read-back). Three alternating rounds, spin layout on,frac0.6,cache2.5GiB,no MTP, no probes: DMA46.783/46.217/46.437 vs fetch48.051/48.075/47.801ms; fetch slower by1.267/1.859/1.364ms (median1.364). Text hashes differ between arms, so workload identity is not established.

Reverted: copy_stream fetch API, its test, the dma_miss branch and the CUDA source removed; release build and58 unit tests pass (5 ignored), git diff clean of the experiment. Patch retained at harness/m41/patches/ (gpu-fetch-copy-stream.patch, gpu-fetch-dma-miss.txt, fetch.cu) with results in harness/m41/results/gpu-fetch-ab.json.

Interpretation: the per-projection, per-expert launch shape costs more than it saves. This is not evidence against Strata-style kernel staging, which issues one fetch_blobs launch per group over device-side descriptors, then rebases pointers. A fair retry needs one batched launch per layer driven by device-visible descriptors, not three launches per expert. No default changed.

## Why Strata's kernel staging wins and naiwa's loses

Strata source: DMA mode (`pcie_mode` 0) issues cudaMemcpyAsync per expert on a private copy stream, then cudaLaunchHostFunc on that stream raises a mapped flag (fetch_dma in verify.cpp); the compute stream spins in wait_flag_ge(m_flagB_) for it. Kernel mode (pcie_mode 2) is called with n=0, so fetch_dma raises flagB immediately, and the GPU pulls the blobs itself with fetch_blobs, stream-ordered, then rebase_ptrs. Kernel mode therefore wins by removing a host-callback-mediated cross-stream barrier, not by transferring faster. Measured within Strata with identical text: 37.15/37.19/37.23 vs 39.43/40.87/39.23 ms/token.

Naiwa has no such barrier: memcpy_htod_async on the copy stream plus cudaEventRecord, and the compute stream waits on that event in stream. There is no host flag and no spin. Substituting a kernel removes nothing and instead takes SMs away from the expert GEMMs. Measured: per-projection fetch +1.36 ms/token, batched mapped-descriptor fetch +1.77/1.67/1.26; both reverted. Microbenchmark confirms the fetch kernel reaches 0.97-1.00 of memcpy bandwidth, so the loss is contention, not transfer rate.

Second asymmetry, from the same runs: staged share 23.8% of naiwa selections vs 14.7% of Strata's (ROUND_STATS vs the engine's decode hit-rate line). At 480 selections/token that is ~114 vs ~71 experts/token, i.e. ~1.6x the bytes over PCIe (~240MB vs ~150MB at ~2.1MB per expert). Strata instead leaves 38.8% of selections to its CPU pool with its host thread participating, and its CPU wall time overlaps its GPU work. Naiwa stages more because frac0.6 trades CPU exposure for PCIe bytes, and PCIe is the binding chain.

Consequence for the next experiment: the lever is the staged byte volume (placement/hit rate, or making the CPU share hidable), not the transport mechanism. Adopting kernel staging in naiwa would first require cutting what crosses PCIe.
