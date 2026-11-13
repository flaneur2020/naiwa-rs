# M42 diagnostic repeatability attribution

**Status:** completed diagnostic experiment; not a performance comparison.
**Parent:** M42 / attribution harness.

## Configuration and evidence

`repeatability.json` and `repeatability-no-lookahead.json`: identical binary/config except --no-lookahead, frac .25, deterministic combine, physical layout, five workers plus caller, 20ms spin, no drafting, async CPU activation disabled. Three AB/BA restart rounds, one greedy 128-completion-token request/process. Both named arms are the SAME baseline. Six processes per condition. See results/repeatability.json for complete hashes, first differing records and per-field divergence positions. Raw outputs: /tmp/attr-repeat-02 and /tmp/attr-repeat-nolook-01.

The first attempt (/tmp/attr-repeat-01) encountered JSON interleaving with the normal logger; do not use it as a complete trace comparison. Corrected instruments use independent work.jsonl files. Residency snapshot is now taken inside the SAME state lock as hit/miss classification (the earlier version acquired a second lock after classification and could capture a newer state).

## Results

| condition | pair 1 first work mismatch | pair 2 | pair 3 | all six full traces equal |
|---|---|---|---|---|
| lookahead on | index 5, layer 5 tick 1 | index 5, layer 5 tick 1 | index 98, layer 2 tick 3 | no |
| lookahead off | none | none | none | yes |

Each trace has 6192 records. Indices are zero-based, and ticks are per-layer local counters, NOT generated-token indices. Records include all logged layer calls; no invented decode-only alignment.

With lookahead on, first differing fields are residency, GPU slots/weights, CPU IDs/weights. At that exact record router IDs and weight bits still match. First subsequent router differences:

| pair | first differing weight bits | first differing IDs |
|---|---:|---:|
| 1 | 6 | 10 |
| 2 | 6 | 10 |
| 3 | 99 | 99 |

Five of six lookahead-on texts have SHA256 e19ca091d184a0157f86d5f8f7670eae39e142cfaf1548f085f51111717c7e52; one has 627162399b8e065681cca3e1493c250c578a843c87669202bcc81bad3cb9ea89. ALL six lookahead-off texts have SHA256 5d35365870b2f2495e87d5471eca60ed620b8fa7ddd5bbc0bacd94f6611777aa. More importantly ALL SIX work traces match in IDs, exact weight bits, residency and final assignment.

## Interpretation / source boundary

- `src/moe/stages/preload.rs::Preload::submit` schedules a background job predicting the next layer using this layer's activation.
- `src/moe/cached_layer.rs::prefetch_predicted` calls admit/replace and publishes a preload claim after the copies. This intentionally mutates residency asynchronously.
- `src/moe/stages/route.rs::Route::submit` classifies against whichever residency exists when it acquires the state mutex.

Observed order is consistent with asynchronous prefetch arriving before/after classification, then CPU/GPU arithmetic-path differences affecting later routing. Evidence establishes a lookahead-dependent repeatability problem in this configuration, not that copies are corrupt or that admit/replace is individually buggy. Different CPU/GPU reduction arithmetic can legitimately diverge numerically. No request-correlated prefetch commit log or activation identity test was collected, so an exact mutation cannot yet be named.

## Validation and next gates

Python 8 tests pass; Rust 58 passed, 5 ignored; release build passed. Independent trace sink tested in 12 real model processes. These diagnostic timings are NOT performance evidence. Do not disable lookahead by default: prior clean runs showed its benefit.

Use no-lookahead only as a repeatable diagnostic control for scheduling experiments; do not compare its TPOT against a lookahead-on arm. Next add request/step-correlated prefetch proposal/commit events if pursuing the on-path race. To establish critical-path overlap, collect actual CPU/result and GPU timestamps in that repeatable control before another scheduling change. Neither matching work records nor matching text proves activation/byte identity.
