# CPU result publication boundary

Status: host publication measured; GPU consumption remains unmeasured. Parent: buffered host scheduling/T4c-b2.

`Executor::submit_result_tagged` optionally tags CPU jobs with layer/tick, logging buffered events immediately before result publication and after it returns. Existing callers use untagged wrapper; default computation, error handling and result format unchanged. Includes result-ring reserve, host payload copy, release sequence publication and event-record overhead; NOT device consumption. A failed/drop publication would still return, so source/engine errors must also be checked for correctness.

Three AB/BA/AB diagnostic rounds using cpu-first-buffered.json, no-lookahead frac .25 spin5+caller20ms/no drafting, one128-token request/process, no heavyweight work logs. Raw /tmp/attr-cpufirst-publish-01. Per-phase filtered tick>=3 (layer-local, NOT guaranteed decode-only),5558 tasks/arm/round, no dropped events. JSON summary cpu-first-publish.json.

| round | base mean/p99 us | CPU-first mean/p99 us |
|---|---|---|
|1|1.454 / 2.224|1.415 / 2.344|
|2|1.450 / 2.335|1.448 / 2.225|
|3|1.399 / 2.205|1.502 / 2.274|

All six maxima <57us, zero >=100us. Within this condition there is no sign of large publication/backpressure stalls, so prioritize downstream GPU-consumption/dependency analysis, not optimizing host result memcpy. Does not generalize to MTP or other workloads, and is not a TPOT gain estimate.

MTP previous regression audit: r3 last periodic logger sample base accept67.1%/len2.34 vs candidate61.0%/len2.22. The5s logging windows are not per-request aligned; other windows differ and some end before idle. This is a clue only; it cannot establish that acceptance caused the +1.352ms/token regression. Need real request/verify-step proposed/accepted counters. Do not compare these window rates as complete experiment acceptance totals.

Validation: Rust58 passed/5 ignored, Python9 passed, diff check clean. Remaining gates: GPU consume timing+clock bridge+request IDs, MTP per-step counts, chat. Defaults unchanged.
