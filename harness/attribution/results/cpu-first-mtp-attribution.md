# MTP count-versus-amortized-cost attribution

Status: count/cost decomposition established in NEW diagnostic runs; historical clean round3 remains uninstrumented. Parent: T4c-c/M42. CPU-first DEFAULT OFF.

## Instrument / scope

Fork speculative/driver.rs optionally buffers [proposed,accepted,keep_len] by sequence id and writes one JSON record when finish_verified_step marks sequence terminal (NAIWA_MTP_TRACE_FILE). No per-step file formatting; allocation/locking and terminal serialization remain diagnostic overhead. Captures verification outcomes, NOT bootstrap/drafting/target GPU durations, request IDs across servers, cancellation cleanup, or GPU identity. One sequential request at a time makes terminal sequence order correspond to request order here; harness checks terminal record count equals requests. Aborted/cancelled sequences can leave pending diagnostic vectors until process exit. Default off; bounded experiment18 requests, not long-running telemetry.

Config cpu-first-mtp-acceptance.json: Rayon, frac .4, lookahead on, fixed greedy MTP2, deterministic combine, AB/BA/AB, three requests/process,128 completion tokens, no failures/retries. Raw /tmp/attr-cpufirst-mtp-accept-01. Every process has3 terminal sequence records. Exact per-request sums/histograms+full text hashes archived cpu-first-mtp-acceptance.json; algebraic components cpu-first-mtp-decomposition.json. Timings are diagnostic SSE client timings, NEVER mix with clean estimates. Actual workloads/text differ.

## Exact accounting

Let D=client first-content→DONE milliseconds, N=number of completed verify outcomes, C=D/N. For fixed127 output intervals:

Delta TPOT = [(N1-N0)*(C0+C1)/2 + (C1-C0)*(N0+N1)/2] /127.

This is an exact symmetric accounting identity, NOT causal standardization for different accepted tokens, histories, row counts or expert workloads. C includes drafting, other execution, bootstrap/terminal boundary and transport; it is NOT pure verify-kernel time. N labels completed outcomes, not all model forward calls. Counterfactual same-work cost is not identified.

## Observed regression explained at the accounting level

Round2/request2:

| |base|CPU-first|
|---|---:|---:|
|verify outcomes|55|58|
|accepted/proposed|64.5%|58.6%|
|client decode ms/outcome|90.98|90.34|
|TPOT ms/token|39.40|41.26|

Delta TPOT +1.858 = count component +2.142 plus amortized-cost component -.284 ms/token. The regression here is fully accounted for by extra verification count overcoming lower amortized cost. Round2/request3: +.019 = count +1.417 and cost -1.397. Therefore worse TPOT can coexist with lower cost per completed verification, and is not by itself evidence that CPU-first slows execution.

All9 corresponding diagnostic request pairs have lower C with CPU-first, but count changes -4..+3, adding/subtracting ~.71ms/token per outcome around these ~90ms cycle costs. With equal N (r1/request3,r2/request1,r3/request2), TPOT deltas -.427/-.909/-.707. This is descriptive corroboration, NOT same-work causality (acceptance positions/text/expert assignments can differ even when counts match).

Historical clean MTP round3 +1.352ms/token had no terminal outcome trace. Do NOT claim this newly measured mechanism proves the cause of that historical regression. Periodic logger rates remain insufficient. Need a clean-compatible acceptance snapshot/protocol or independently repeated count-aware runs to assess policy consistently.

## Integrated interpretation

Earlier repeatable no-lookahead runs showed earlier CPU submission and positive clean TPOT; publication mean1.4–1.5us, so host publication is not the large delay there. New MTP evidence identifies a distinct execution-count confound that can reverse output-token metrics. Root of acceptance/work divergence not fixed: lookahead timing changes residency, CPU/GPU arithmetic paths can shift target routing, and drafter atomic aggregation remains nondeterministic. Cannot assign each acceptance difference to one of these without negative controls.

Next: isolate MTP with lookahead off, audit drafter determinism, then GPU consume/clock-correlation for critical path. Default adoption/chat acceptance still open. Validation release build/check pass; Rust58 passed5 ignored; Python10 passed. Debug cargo check twice timed out compiling dependencies; release cargo check succeeded.
