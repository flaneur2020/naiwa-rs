# Scheduling attribution execution checklist

Parent: docs/worklogs/m42-token-thread-sync-reads.md. No default configuration changes. No candle operator changes.

- [x] T1 Repeatable diagnostic control: no-lookahead, frac .25, no drafting, deterministic combine, physical 5+caller, spin 20ms. Six work traces identical (results/repeatability.json).
- [x] T2 CPU-first staging hook (default off, NAIWA_CPU_BEFORE_DMA=1): select exactly the same staged expert set; submit remaining CPU work before staging DMA; preserve hit order, staged bytes, output addition, fences. Default off.
- [x] T3 Gates (tests/oracle; three pairs of 18480 records identical): release tests; frac0/no-lookahead oracle first line exact; separate diagnostic paired runs verify complete work records. If work mismatch, stop attribution and locate first divergence.
- [x] T4 Control-group clean AB/BA/AB (-.907/-1.219/-.835 ms/token; see results/cpu-first.md): >=3 restarted rounds, 128 completion tokens, actual server usage denominator, per-round median; content-gap tails are SSE, not decode steps. Keep only reproducible mean benefit or stable tail benefit without mean regression. Revert unsuccessful implementation, archive evidence.
- [x] T4b Second prompt and lookahead-on verification at experimental frac .25/no drafting: second prompt clean paired -1.580/-.991/-1.258 ms; 3 diagnostic pairs of 18336 records identical. Lookahead-on -.114/-.601/-.620 ms, text differs (policy comparison only). See results/cpu-first-followup.md.
- [x] T4c-a Default Rayon/frac .4/lookahead-on clean checks: no drafting -.574/-1.126/-1.021 ms; fixed MTP2 -1.003/-.891/+1.352 ms. MTP gate NOT passed, defaults unchanged. See results/cpu-first-default-and-timeline.md.
- [x] T4c-b1 Buffered host/CPU events without hot-path JSON: ~26us earlier submit,24–32us earlier worker, DMA enqueue11–13us later; no dropped events. See results/cpu-first-buffered.md. Old ~300us live-logging figure discarded.
- [x] T4c-b2a Result publication buffered host boundary: mean1.4–1.5us/p99~2.2–2.4us, no >=100us across six captures; results/cpu-first-publication.md.
- [ ] T4c-b2b GPU-clock correlation + GPU consumption + request/verify-step identity; buffered host-only events do NOT measure exposed GPU waits.
- [x] T4c-c1 MTP per-sequence step counts/proposed/accepted captured, exact count-vs-amortized-client-cost decomposition. New observed regression +1.858 = count+2.142/cost-.284ms; results/cpu-first-mtp-attribution.md. Historical clean r3 cause NOT proven.
- [x] T4c-c2a MTP no-lookahead: all9 pairs match full ordered acceptance trace+text; separate clean deltas -.900/-1.021/-.561ms. See results/mtp-nolook-control.md.
- [ ] T4c-c2b Lookahead proposal/commit/classification/request IDs; universal drafter determinism not established. Clean-compatible acceptance measurement and multi-turn chat remain.
- [ ] T5 Mandatory async activation channel lifecycle: explicit read/skip acknowledgements, no advisory stale drops; bitwise inputs, oracle, determinism before benchmarking. Independent experiment; do not bundle with T2.
- [ ] T6 Return to lookahead-on production settings; correlated prefetch proposal/commit/classification events. Do not infer performance from diagnostic runs.
- [ ] T7 Decide device-side resident dispatch only after above evidence; residency lifetime and stale-map negative control required.

MTP depth: explicitly fixed when drafting enabled. With drafting off CLI forbids --mtp-n-predict without --mtp; omit with documented exception, never enable drafting just to supply it.

- [x] Replacement budget1 follow-up: second prompt no drafting, two prompts MTP2, original CPU order all3-round clean wins; independent MTP count/cost report. See results/fill-budget-validation.md.
- [ ] Budget1 chat/tail/TTFT and correctness adoption gates; GPU consumption/request correlation to separate transfer contention from placement. Defaults unchanged.

- [x] Single-sequence argmax pipeline rejection identified; opt-in existing GPUtop1 path wins two prompts and no-lookahead text control. See results/single-gpu-sampling.md.
- [ ] Single-GPU-sampling adoption: oracle/ties/NaNs/fallback tests, MTP2/chat, second no-lookahead identity and fresh Strata comparison. Default off.
