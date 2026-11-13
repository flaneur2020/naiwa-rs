# T4c-b buffered CPU/host scheduling capture

Status: host-clock portion completed; GPU/result-publication correlation remains OPEN. Parent: TODO.md and M42 S10.

## Instrument

Default-off `NAIWA_SCHEDULE_FILE` records numeric host-monotonic timestamps, layer-local tick, layer and static phase in a bounded mutex-protected vector. No per-event formatting/file I/O; a writer drains after 300ms of inactivity. Capacity131072; drop counts and flush markers explicit. Runner waits2s after requests and rejects missing artifacts. Analyzer rejects empty, dropped, duplicate, incomplete phase sets. Event metadata alone does NOT provide request/verify-step identity or GPU-clock calibration. Lock/timestamp overhead still perturbs; idle-triggered writer is not a guaranteed request-boundary flush and may run during long stalls. Failed/killed runs may lose buffered tail. Fused compute-return only; generic fallback lacks return event and is rejected by analyzer.

Initial compile failed (serde derive unavailable); shell incorrectly proceeded with old executable: /tmp/attr-cpufirst-buffered-01 has no event files and is INVALID. No results used. Fixed compile and added missing-artifact gate. New run /tmp/attr-cpufirst-buffered-02 is independent, not a request retry. Tests Rust58 passed/5 ignored, Python9 passed.

## Experiment / measured boundaries

Three restarted AB/BA/AB rounds, no-lookahead, frac .25, spin physical5+caller20ms, no drafting, one128-token request/process. WORK_TRACE and CPU_TIMELINE disabled to eliminate asymmetric serialization. Six processes each have6192 route/dma/submit-return phases and5654 matching CPU submit/start/return events, dropped0. All intervals on one process's Instant clock. Explicit filter local tick>=3 (NOT a guaranteed decode window). Diagnostic TPOT not used. Results cpu-first-buffered.json, config cpu-first-buffered.json (parent directory).

| round/arm | route→submit mean us | route→worker | route→compute return | worker→return | route→DMA enqueued |
|---|---:|---:|---:|---:|---:|
| 1 base |37.5|48.3|554.3|506.1|21.7|
| 1 early |11.6|24.4|536.5|512.1|34.2|
| 2 base |37.6|52.2|564.2|512.0|21.8|
| 2 early |11.6|20.5|524.3|503.8|32.9|
| 3 base |37.8|50.4|563.3|512.9|21.9|
| 3 early |11.7|24.4|532.0|507.6|33.6|

CPU submit shifts earlier ~26us/layer, worker earlier24–32us, compute return earlier18–40us. Staging DMA enqueue shifts LATER11–13us, the tradeoff. Worker→return does not consistently shrink: supports scheduling reordering, not faster CPU arithmetic. Unlike previous ~300us figure, no hot-path work JSON exists in this capture. Still cannot multiply layer differences into a TPOT prediction, nor equate CPU return with GPU publication/consumption. No same-work trace in these timing runs; matched identity evidence belongs to separate diagnostics.

## Remaining

Add result-publication boundary, GPU event/capture correlation and request/step IDs; inspect MTP third-round regression with real per-step accepted/proposed counts. CPU-first stays opt-in; no default/chat adoption from this host-only capture.
