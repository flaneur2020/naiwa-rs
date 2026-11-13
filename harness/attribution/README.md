# Performance attribution harness

Stdlib-only. Run from repo root. This is a separate, reproducible path; old milestone scripts are retained as historical evidence, not silently reinterpreted.

## Restarted A/B

```
python3 harness/attribution/run.py config.json --out /tmp/attribution-run-001
python3 harness/attribution/analyze.py report /tmp/attribution-run-001/manifest.json
python3 -m unittest discover -s harness/attribution -p 'test_*.py' -v
```

Config shape (use actual engine argv, no shell fragments):

```json
{
  "mode": "clean",
  "rounds": 3,
  "requests": 3,
  "drafting": "off",
  "env": {"MISTRALRS_MOE_DET_COMBINE": "1"},
  "body": {
    "messages": [{"role": "user", "content": "Write a detailed description of how a database index works."}],
    "max_tokens": 128, "temperature": 0, "top_p": 1, "seed": 0,
    "enable_thinking": false
  },
  "arms": [
    {"name": "base", "cwd": "/path/to/repo", "url": "http://127.0.0.1:15951", "argv": ["/path/to/naiwa-rs", "serve", "...actual arguments..."]},
    {"name": "candidate", "cwd": "/path/to/repo", "url": "http://127.0.0.1:15951", "argv": ["/path/to/naiwa-rs", "serve", "...actual arguments..."]}
  ]
}
```

- At least three **restarted** rounds; AB / BA / AB. Each cell uses its process's request median; paired differences then summarized by median. Request position is stored and raw per-request results preserved: first-request and cache-reuse effects must be examined separately, not called independent repetitions.
- New output directory required; failures persist and abort; no request retries. Readiness polling is not a performance request. Already-serving endpoint rejected. Only the launched process group is terminated, never global pkill.
- Ambient `NAIWA_*`/`MISTRALRS_*` removed; desired variables must be explicit in config. Clean mode rejects common diagnostic variable names unless explicitly allowlisted. This heuristic cannot detect all instrumentation: inspect argv/builds too.
- No-drafting (`off`) omits `--mtp-n-predict`: naiwa CLI requires `--mtp` if depth is supplied. With drafting `on`, depth must be explicit. This declaration does not verify that an arbitrary external engine disabled all drafting methods; retain engine logs and inspect actual draft counts.
- No completion-count fallback: server must return `usage.completion_tokens` and exactly 128. Short output or malformed/truncated stream is a failed run. The request asks for streamed usage; unsupported servers need an explicit adapter, not chunk counting.
- TPOT denominator is `completion_tokens - 1`. Capture is SSE client first content to DONE, not engine-internal decode. TTFT includes request/prefill; neither is a pure kernel timer.
- Full text SHA256, executable SHA256 when resolvable, git HEAD/status/tracked-diff SHA256, content text, every content chunk timestamp, usage, finish reason, argv, selected environment are saved. Untracked source contents and fork dependencies are not hashed automatically; archive them separately for full reproducibility. SSE chunks/gaps are **not** tokens or decode steps, especially with drafting.
- `performance_eligible` only checks complete clean >=3 paired rounds. Text or work mismatch prevents causal same-work claims, even when end-to-end results remain measurable. Full text equality is necessary, not sufficient, for identical MoE work.
- Config/logs may contain prompts, local paths, secrets in argv; treat output as private. No network/webhook uploading.

## Diagnostic work identity (naiwa)

Use a separate config with `mode: diagnostic` and `NAIWA_WORK_TRACE=1`. The runner assigns a unique `NAIWA_WORK_TRACE_FILE` (`work.jsonl`) per process to prevent logger interleaving. For manual launches set this path explicitly; stderr fallback may interleave. Never compare its TPOT with clean arms. The disabled probe performs no additional device reads. Each `MOE_WORK` JSON line records:

- layer and per-layer tick, rows;
- ordered router IDs and exact f32 weight bits;
- slot→expert residency before staging;
- final GPU slot/weight bits and CPU expert/weight bits after staging.

```
python3 harness/attribution/analyze.py routes left/work.jsonl right/work.jsonl
```

Compares ordered records and reports the first mismatch, including the complete differing records. Empty/missing traces never pass. Duplicate layer/tick rejected. Compare **one process, identical request sequence** per file. Tick is local to the layer, not a global request/step ID; no filtering to decode or cross-request realignment is guessed. Prefill can be included. Different earlier request history invalidates alignment. No activation checksum, transfer-byte identity, GPU result identity, or clock correlation is proven by this probe. Stage slot assignments are ordered; different slots are correctly flagged even if expert IDs match. The intentionally conservative verdict is not a throughput acceptance gate.

## CPU task and counter boundaries

```
python3 harness/attribution/analyze.py cpu engine.log
```

Parses individual `CPU_TASK` records (enable `NAIWA_CPU_TIMELINE=1` in diagnostic mode) and reports submit→worker and worker→return distributions. The latter includes activation acquisition, not just fused arithmetic. Parses `ROUND_STATS` by differences between cumulative pass counts, never by dividing by the cumulative counter. First window is marked boundary-contaminated, not silently discarded. Neither accumulated job wall time nor these percentiles measure GPU-exposed CPU waiting. Rechecked `/tmp/step-half-ret0.log`: mean 30.33 ms/pass excluding first window, correcting the prior erroneous ~9.44 calculation.

## GPU timeline (nsys CSV)

Export `nsys stats --report cuda_gpu_trace --format csv capture.nsys-rep` to a unique file, then:

```
python3 harness/attribution/analyze.py gpu trace.csv \
  --start-ns 123000000 --end-ns 456000000 --steps 127
```

**Start/end must be an explicitly identified decode-only window on the capture's GPU clock**, and steps must come from actual engine decode steps (not completion tokens under MTP). The analyzer never guesses a suffix from client wall time or silently drops outliers. Clipped intervals, multi-stream union, idle, class sums, class unions and exclusive occupancy are separate. Exclusive(class) = union(all) − union(without class). Nested overlapping events cannot mislabel the event actually ending a busy component. Gap classes label neighboring operations, not causal blame. Global idle is not compute-stream idle; neither exclusive occupancy nor idle is a rigorous optimization speedup ceiling. Copy classification is based on nsys memcpy names; memset and unrecognized activities count as kernel/other, so inspect names if present.

## Current limits / next instruments

No automatic engine-to-GPU clock correlation, request/verify-step IDs, activation checksums, acceptance distribution, physical transferred bytes, or Strata event adapter yet. These are explicit gaps, not inferred fields. Work trace is scheduling-perturbing diagnostics, not a clean benchmark. CPU RET_DBG windows remain aggregate means; do not label them per-task percentiles. ROUND_STATS cumulative `rounds` is not the denominator for each reset window; use window length and treat first/prefill/boundary windows separately.

## Buffered host schedule capture

Set `NAIWA_SCHEDULE_FILE` in diagnostic config; runner assigns per-process schedule.jsonl and waits for idle writer. Do NOT also enable WORK_TRACE/CPU_TIMELINE for timing attribution. Analyze `python3 harness/attribution/analyze.py schedule path/schedule.jsonl`. Numeric host events use one monotonic clock, bounded capacity with drop accounting; output flushes after300ms inactivity. Initial allocation/locking and writer still perturb timing. No request or GPU-clock correlation; tick>=3 analysis filter is not guaranteed decode-only. CPU fused return is not result publication; generic fallback is rejected as incomplete. Missing/dropped/incomplete events invalidate analysis. See results/cpu-first-buffered.md.

## MTP acceptance and count/cost report

`python3 harness/attribution/analyze.py mtp RUN_DIRECTORY` requires a complete diagnostic config/manifest with per-process mtp.jsonl terminal records and request captures. Enable NAIWA_MTP_TRACE_FILE in diagnostic config; runner assigns unique paths and validates terminal count, unique sequence IDs, legal proposed/accepted, monotonic keep_len. Supports sequential single-client requests only. It reports full step-sequence hashes/histograms, matched request-position text equality and exact symmetric count/cost accounting. Cost is client decode time divided by completed verification outcomes, NOT verify kernel duration. Missing/aborted requests invalidate analysis; counts cannot be backfilled from periodic logger percentages. Never merge diagnostic timing with clean A/B.

Reference negative control and clean follow-up: results/mtp-nolook-control.md. GPU/request-clock correlation and concurrent-client association remain unsupported; equality of acceptance sequence/text does not prove activation/byte identity.

## Three-arm engine refresh

Runner also accepts3 arms with rotated ABC/BCA/CAB order. Use strata-three-arm.json for current no-drafting baseline/CPU-first/Strata comparison; strata_adapter.py materializes an owned per-process config via {run_dir}. Check native Strata logs for actual zero drafting (spec4 alone does NOT mean drafting). Analyze report supports three-arm medians/paired differences. Practical engine comparison does not align cache capacity, text or internal timers. See results/strata-three.md.

## Cross-engine byte ledger (diagnostic)

Run engine-ledger.json then `analyze.py ledger RUN`. Strata adapter requires opt-in STRATA_WORK_LEDGER_FILE and uses instrumented build/strata; source patch in patches/strata-work-ledger.patch. Artifact validated for3 serial requests/128decode positions/48layers; no concurrent/batch support. Naiwa byte events share buffered schedule (capacity1048576) and are repeatable phases; only timing phases must be unique. Compare selected PCIe payload, not memcpy-only counters; windows still not request-aligned on naiwa. See results/engine-ledger.md for omissions and failure evidence.

## Fill lifecycle

`python3 harness/attribution/fill_lifecycle.py schedule.jsonl` pairs cache H2D enqueue/publication by(layer,slot,expert), closes generations on successful fill/promotion publication, tracks resident route uses, and separates right-censored tails. Requires expert/slot-aware instrumented build. Route use is NOT GPU consumption; borrow/error paths and request boundaries unsupported. Diagnostics and clean replacement-budget configs/results: fill-budget*.json / results/fill-budget.md. No benefit inferred from diagnostic timings.

## Step GPU ledger

`python3 harness/attribution/step_costs.py RUN` validates raw Strata GPU-window stamps and returns per-step disjoint compute-stream partitions plus separate host summaries. Requires extended Strata patch and profile enabled; never compare these intervals by subtraction with device-wide nsys unions. Naiwa summaries currently host-only; matching GPU partition remains open. See results/engine-step-costs.md.

## Naiwa stream partition (diagnostic only)

engine-partition.json enables reusable CUDA events with NAIWA_GPU_PARTITION_FILE (runner assigns per-process path). `compare_partitions.py RUN` validates ordered phases/drop markers and selects full48layer rows1 sequences, reports rolling route-to-route intervals vs Strata profile. No nai​​wa request IDs; tail incomplete groups censored. Event allocation/query cost still measurable: partition-overhead.json calibrates. v1 had~5.9ms perturbation; v2~1ms median and cannot be corrected per bucket. See results/structural-attribution.md before interpreting any delta. Non-MoE bucket includes host/framework work and head, not pure dense kernels.

## Model interior probe

model-partition.json compares optional NAIWA_MODEL_PARTITION_FILE with no-probe arm; fork callback layer/mixer/router-expert/shared endpoints. model_costs.py RUN STRATA_RUN validates model groups and exports coarse brackets; no cross-session causal subtraction. Default callback absent, GPU events not allocated. See results/model-attribution.md for measured overhead (~.5ms median), boundary caveats and reverted shared-early test.

## Serial conversation validation

Optional chat_followups must contain requests-1 strings; runner appends actual assistant full text and next user message, persists request-N-body.json. Different outputs create different later inputs: policy comparison only, not fixed-transcript same-work proof. Capture now retains SSE logprobs payloads for fallback parity checks. See results/validation-followup.md for TTFT regressions and unpassed adoption gates.
