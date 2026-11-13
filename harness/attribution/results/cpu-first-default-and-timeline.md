# T4c partial: default thread pool / MTP2 / diagnostic timing audit

Status: experiments completed; correlated GPU critical-path attribution STILL OPEN. CPU-first remains default off. Parent: attribution/TODO.md.

## Clean default-thread-pool checks

Both arms Rayon default (no spin env), frac .4, lookahead on, deterministic combine. Three restart AB/BA/AB rounds, three requests/process, 128 completion tokens. No request failures/retries. Configs cpu-first-default-{nospec,mtp2}.json; full manifests and SSE tails in results/cpu-first-default-*-{manifest,tails}.json. Raw /tmp/attr-cpufirst-default-{nospec,mtp2}-01. No cross-session absolute comparison.

| mode | round | base median | CPU-first | delta ms/token |
|---|---:|---:|---:|---:|
| no drafting | 1 | 48.525 | 47.952 | -.574 |
| no drafting | 2 | 48.875 | 47.749 | -1.126 |
| no drafting | 3 | 48.488 | 47.466 | -1.021 |
| MTP depth 2 greedy | 1 | 41.792 | 40.789 | -1.003 |
| MTP depth 2 greedy | 2 | 41.393 | 40.501 | -.891 |
| MTP depth 2 greedy | 3 | 40.601 | 41.953 | +1.352 |

No-drafting median paired delta -1.021 ms/token; p95 SSE gap improves all rounds, p99 round3 slightly regresses 65.167→65.661. MTP2 median paired delta -.891 but round3 mean regresses materially: NOT a stable MTP adoption result. MTP SSE p95 worsens rounds1/3, p99 improves all three. Neither condition has stable text across corresponding request positions. These are scheduling-policy end-to-end comparisons with lookahead-dependent work differences, not same-work proofs. MTP logger confirms built-in assistant depth2; five-second aggregate acceptance logs exist but are not per-request/step acceptance distributions. Do not explain the round3 regression as drafting acceptance without matching step evidence. MTP TPOT remains per output token (127 intervals), not per verify step.

## Separate CPU timeline diagnostic

Config cpu-first-timeline.json: no-lookahead, frac .25, spin layout20ms, one request/process, three pairs; CPU_TIMELINE + WORK_TRACE. Each paired trace has 6192 records and matches. CPU_TASK count 5654 per process. Raw /tmp/attr-cpufirst-timeline-01. Parsed results cpu-first-timeline.json and cpu-first-timeline-filtered.json.

All-record mean route→worker is ~340–345us base vs ~26–31us CPU-first. HOWEVER this is NOT an estimate of production CPU-start improvement. Source-order audit finds a diagnostic confound: MOE_WORK serialization/file write occurs AFTER staging and BEFORE baseline CPU submit, but AFTER CPU-first submit. Thus instrumentation adds asymmetrical startup delay. Filtering rows=1/tick>=3 cannot remove that perturbation (filtered difference remains ~300us). The filter is explicitly layer-local, not a guaranteed decode-only boundary. Worker→return also includes activation acquisition; completion timestamp is not GPU-visible publication/consumption.

Conclusion: discard the magnitude as mechanism evidence. Clean benchmarks have no WORK_TRACE and are not affected by this particular logging confound. Work equality diagnostics remain useful as perturbed workload-identity observations, but cannot prove identity of separate clean runs. No GPU clock correlation collected, so T4c critical-path gate remains incomplete.

## Next work

Use a bounded buffered event recorder: cheap timestamps/IDs in hot path, serialize only after request/completion. Align CPU submission/start/result-publication and GPU result-consumption events with explicit request/step IDs and GPU-clock calibration. Disable heavyweight work tracing in timing captures, keeping identity runs separate. Then investigate MTP regression with actual per-step proposed/accepted counts, not aggregate logger rates. Multi-turn chat verification still outstanding; do not enable default based on no-drafting-only benefits.
