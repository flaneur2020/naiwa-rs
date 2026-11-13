# Cross-engine transfer/work ledger

Status: collected3 diagnostic restart pairs; comparable payload definitions, NOT aligned request windows/causal gap decomposition. Parent: M42 S16.

## Implementation

Strata `STRATA_WORK_LEDGER_FILE` opt-in in core/verify.cpp: buffer per-layer run() row after pool returns, with absolute position, nrows, GPU entry count, resident/staged distinct-group counts, staged_groups*actual native layer blob bytes, pool callback wall time. Existing request profile_report drains JSONL and boundary marker; formatting/I/O outside hot layer. Counts are final plan, not preclassified IDs. GPU entry count includes resident+staged rows; CPU entries derived10*nrows-gpu_entries only valid here single GPU/no helper. Kernel staging payload is still PCIe traffic, even though fetch_dma sees n=0. Counter is selected payload, not hardware byte transactions. Existing GPU stamps plus DECODE_TIMING retained for descriptive stages; pool callback not comparable to naiwa worker compute. All-resident/slot/batch/pipeline paths unsupported. No buffer capacity/drop enforcement yet; controlled128token3request runs only. Empty-plan fallback is not explicitly tagged; current requests have valid published plan. Output-open errors warn, harness detects missing artifact. Env alone does not ensure report callback: enable VERIFY_PROFILE too.

naiwa existing buffered schedule extended with staged_h2d and cache_fill_h2d byte events, recorded after successful queue; cache_fill covers upload_in_place only, combines lookahead/admit/replace/restoration callers, not solely lookahead. No fallback-owned upload/promotion D2D counter yet. Tick is sampled layer-local at enqueue, not immutable triggering request ID. Schedule buffer raised131072->1048576 due repeat-request artifacts exceeding capacity; allocation/lock cost diagnostic. File schema additive bytes field (zero on timing events). Parser separates repeatable byte events from singleton phases.

`analyze.py ledger RUN` validates Strata3request x128consecutive positions x48distinct layers (rows1; rows>1 prompt excluded) and naiwa flushed/no-drop schedule before summarizing. Request mapping is positional serial-only. Strata template adapter selects build/strata for ledger runs, original engine/strata unchanged. SHA256 recorded engine-ledger-provenance.json; source patch patches/strata-work-ledger.patch. CUDA build passes, HIP/SYCL NOT tested. No default changes.

## Collection and failures

Raw /tmp/attr-engine-ledger-03, config engine-ledger.json;3 AB/BA/AB restart pairs x3requests x128tokens. Diagnostic only. -01 failed duplicate-phase validator (byte events not yet recognized); -02 rejected64678dropped events with old capacity; both invalid, retained. -03 passes schedule and ledger validation; no failures/retries within valid run. `cargo build --release` without cuda failed dependency feature resolution; CUDA release build succeeds. Rust58pass5ignored (correct bin nai​​wa-rs), Python11pass. No oracle gate claimed this turn. Strata patched CUDA build passes; no clean numerical regression gate on patched binary yet.

## Measured payload

Strata values identical across3restarts at each request position:

|request|selected staged MB/decode window|staged groups/layer|CPU entries/layer|
|---|---:|---:|---:|
|1|155.604|1.596|4.135|
|2|141.586|1.455|3.881|
|3|143.551|1.474|3.902|

All have128rows1 windows x48layers. Prompt rows>1 excluded. Per-request pool callback total/window ranges first23.06–23.39ms, reused21.57–23.03ms; NOT exposed GPU wait.

naiwa tick>=3 totals (all3requests pooled, not aligned decode boundary):

|restart|staged H2D MB/48-layer equivalent|cache-fill H2D|sum|
|---|---:|---:|---:|
|1|190.601|169.440|360.042|
|2|190.047|169.552|359.599|
|3|191.726|170.499|362.225|

Cache-fill is ~47% of instrumented naiwa payload; demand staging itself exceeds Strata selected bytes by ~35–50MB; total payload roughly200MB more. This is a candidate at the10ms scale, not proven wasted traffic. Old nsys347MB cannot be subtracted from these360MB totals (different captures, windows and tracing perturbation). Byte payload/bandwidth ratio is NOT inferred performance gain.

## Causal interpretation and next gate

Net cache policies help (clean lookahead-off +.550ms, promotion-off+1.246ms), so simply removing fills is counterproductive. Need trace expert identity, admission kind, displaced expert, publish/use timestamps to count duplicate and unconsumed fills under one request. Need measure whether background fills queue ahead of current demand DMA and whether CPU return becomes exposed because staging and resident compute are not overlapped optimally. Only then select a bounded, opt-in demand-priority/coalescing experiment. Result wait~4.6ms diagnostic exposure remains another large candidate. GPU event/request correlation incomplete; do not add stage timers or assert200MB explains10ms. Strata's waitB includes fetch/graph ordering, unlike naiwa result spin.
