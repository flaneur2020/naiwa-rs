# Per-step Strata GPU partition and separate host comparison

Status: Strata step partition complete for controlled single-stage T1 run(); naiwa matching GPU partition NOT complete. Parent M42 S20. Diagnostic data only.

## Changes and validation

STRATA_WORK_LEDGER_FILE now buffers each window's raw existing GPU %globaltimer stamps (33/layer,48layers+head), pos0,T, host final stream-sync time. Existing layer rows additionally record host wait-to-GPU-doorbell. Dump at existing request profile_report boundary, no per-layer file I/O. Requires STRATA_VERIFY_PROFILE, which changes shared-expert stream behavior (serializes for profiling); cannot be treated as clean scheduling. No host/GPU clock subtraction attempted. No additional GPU stamp launches: exporting existing profile buffer. Raw windows retain every decode step, no percentile exclusion. Controlled128tokens3requests/single-stage only; no universal telemetry bounds/cancellation handling.

step_costs.py partitions monotonic layer endpoints0/17/18/19/20/21/22/23/24 plus next-layer/head boundaries. Missing/reversed stamps, wrong geometry, missing/duplicate host layers,128-step association mismatch and closure errors rejected. Unit tests cover closure/missing stamps. Ledger parser skips typed GPU-window rows. CUDA build succeeds; HIP/SYCL untested. Python14tests pass. Patched engine used build/strata, original engine binary unchanged. Source patch refreshed. Clean output-identity gate remains open; these are diagnostic findings.

Raw /tmp/attr-engine-steps-01, config engine-ledger.json:3 AB/BA/AB restart pairs x3requests/process x128tokens, no retries/failure. Naiwa current CPU-first + budget2 (NOT latest budget1), Rayon/frac .4/no drafting. Budget explicitly retained to compare prior byte ledger shape; do not backfill this to latest candidate performance. Strata no drafting T1. Row>1 prompt windows excluded from GPU/host step ledger. Per-step position association within each serial request. All384decode positions/process valid,9request summaries exported engine-step-costs.json.

## Strata GPU compute-stream partition

Mean over9requests, each128T1windows:

|segment|ms/window|definition|
|---|---:|---|
|dense/attention/router|11.194|layer start through hc-read1+router|
|shared+quant|.884|next stream interval|
|plan wait+copy|.283|shared end to resident begin|
|resident experts|1.538|resident bracket|
|staging wait+fetch+rebase|11.354|resident end to staged begin; NOT isolated transfer|
|staged experts|.522|staged bracket|
|CPU wait|7.965|staged end to CPU result available on stream|
|result copy+combine portion|.386|mapped result copy and layer tail bracket|
|interlayer gap|.032|layer-tail to next layer-start|
|last layer to head|.002|last tail to head-start|
|head|1.185|head bracket|
|TOTAL|35.345|exact unrounded per-window closure|

These are disjoint STREAM intervals including kernel launches/waits/gaps, not all-device interval unions. In particular staging11.354 overlaps CPU service and cannot be called purely PCIe bandwidth cost. CPU wait7.965 is residual after resident/fetch/staged work, not CPU total. Last tail bracket can include combine shared work (see source); names denote boundaries not isolated operators.

## Host ledger, kept separate

Strata host reaches/waits for GPU doorbells12.003ms/window, pool callback22.116, final stream sync1.207. These host phases overlap GPU; do not add to35.345. Profile copy, staging input, sampling, commit/emit and outside-loop overhead not fully captured. Need outer run begin/end to close HOST step wall independently; client TPOT not equal raw GPU span.

Naiwa tick>=3 layer-local host means across3processes (not request boundaries): route_ready->DMA enqueue1.753–1.825ms per48layer windows; DMA->CPU submit return.040–.120; worker->compute return SUM normalized by ALL layer windows24.538–25.261; result publication.088–.096. Conditional job duration ~.656–.668ms; do NOT multiply conditional mean48 ignoring no-CPU layers (previous hypothetical~32ms would overcount). Worker duration includes activation acquisition. Strata pool callback includes planning/CPU calculation and caller participation; NOT same CPU arithmetic boundary/work as naiwa. Thus25vs22 does not establish3ms CPU slowdown or explain residual GPU wait.

## Structural conclusion

Strata itself spends~19.3ms/stream window in staging+CPU-wait brackets. It does NOT win by eliminating waits; wins must be evaluated by critical chain length, work placement and overlaps. Resident+staged arithmetic brackets only~2.06ms while dense/router~11.19; transfer/wait structure remains a large candidate. Do NOT subtract old naiwa nsys copy-union27.37 from Strata staging11.35 or old wait-exclusive4.61 from7.97: different metrics, configurations and captures. Cannot allocate remaining~10ms cross-engine gap numerically yet.

Next required: matching naiwa compute-stream timestamps spanning dense/router, preload event wait, resident, staging event wait, staged, result wait/combine and head with request/step identity. That is the unresolved part of requested same-step decomposition. Also record Strata host whole-run wall and stage-input/profile-copy/commit boundary, independently. Defaults unchanged.
