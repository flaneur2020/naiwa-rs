# MTP no-lookahead negative control

Status: diagnostic acceptance repeatability and clean scheduling benefit verified in this configuration. Parent: M42 / TODO T4c-c2.

## Harness additions

`analyze.py mtp DIRECTORY` now reproducibly validates terminal records, sequence uniqueness, step shape/ranges, monotonic keep_len, request count/output length, then reports each request's proposed/accepted histograms, full ordered-step SHA256, text SHA256 and symmetric exact count/cost decomposition. Runner validates both MTP and buffered schedule artifacts, not just file presence. Sequential single-client terminal ordering is required; concurrent clients/cancellations unsupported. Missing records fail; no best-effort fallback. Terminal trace is diagnostic only.

## Diagnostic negative control

Config cpu-first-mtp-nolook.json, raw /tmp/attr-mtp-nolook-01. Same prior default Rayon/frac .4/greedy depth2 settings except --no-lookahead. Three AB/BA/AB restart rounds, three128-completion-token requests/process. Report cpu-first-mtp-nolook-report.json.

ALL9 paired requests match text and the COMPLETE ordered [proposed,accepted,keep_len] sequence (not merely aggregate acceptance). Position1 has58 outcomes/59.5% acceptance, positions2/3 have59 outcomes/57.6%, repeated across all six processes. Diagnostic candidate TPOT lower for all9 pairs, but do not use those timings as clean performance evidence. Identity covers acceptance trace/text, not activations or physical bytes.

Compared with separate lookahead-on diagnostic runs where counts differ -4..+3 and acceptance can reverse TPOT, this identifies lookahead-dependent execution variation as the observed trigger of acceptance divergence in this workload/configuration. It does NOT prove drafter determinism universally, isolate a particular replacement/copy bug, or retrospectively prove the cause of the historical clean regression. CPU submission reordering alone does not introduce observed acceptance divergence in this control.

## Separate clean A/B

Config cpu-first-mtp-nolook-clean.json, raw /tmp/attr-mtp-nolook-clean-01; archived manifest cpu-first-mtp-nolook-clean-manifest.json. No acceptance/work/timing probes. Three restarted AB/BA/AB rounds, three128-token requests/process, denominator127, no failures/retries.

| round |base ms/token|CPU-first|delta|
|---|---:|---:|---:|
|1|37.794|36.894|-.900|
|2|37.702|36.681|-1.021|
|3|37.536|36.974|-.561|

Median paired delta -.900ms/token. Text matches across arms/restarts at corresponding request positions; global all-text-equal false because first-vs-reused request differs. Separate diagnostic sequence equality cannot prove unobserved clean step equality, but supports the control. Do not compare these absolute times with earlier sessions or with Strata.

## Decision

Retain CPU-first default off. Benefit now reproduced with MTP2 in repeatable no-lookahead control; production lookahead-on acceptance variability remains the adoption barrier, not evidence that CPU-first inherently slows verify execution. Do NOT disable lookahead by default based on these noncontemporaneous conditions. Next correlate lookahead proposal/commit/classification and request/step IDs; GPU consumption/clock bridge and multi-turn chat remain incomplete. Python11 tests pass; no engine source changes this turn.
