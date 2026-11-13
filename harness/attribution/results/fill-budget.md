# Narrowing cache-fill traffic: lifecycle and replacement budget

Status: positive single-prompt clean ablation; default unchanged. Parent M42 S17.

## Lifecycle capture

Numeric buffered events carry expert+slot: H2D enqueue, successful fill publication, promotion publication, resident route use. Publication/use under same layer state lock; serialize ordering in buffer. Promotion closes previous slot generation. Byte accounting paired with publication; no unmatched queues/publications in6process identity diagnostic run /tmp/attr-fill-life-02. Config fill-lifecycle.json has identical arms (repeatability control),3rounds x3requests x128tokens. Captured fraction of H2D bytes published at tick>=3 with no subsequent resident route use before next fill/promotion publication:36.50–38.04%. Tail unused slots right-censored separately, NOT wasted. See fill-lifecycle.json result. No claim of GPU consumption, precise eviction timing, borrowed/error lifecycle or request boundaries. Instrumented policy timing not clean.

## Clean replacement budget ablation

Reuse existing --lookahead-swaps (max replacements per layer per row/pass), no new engine mechanism. Current CPU-first/default Rayon/frac .4/greedy/no drafting, default promotion enabled; same database prompt. Three3-arm rotated ABC/BCA/CAB restarts x3requests x128tokens. Config fill-budget-clean.json, raw /tmp/attr-fill-budget-01. No timing/identity probes; no failures/retries.

|round|swaps2 (current)|swaps1|swaps0|
|---|---:|---:|---:|
|1|48.584|47.747|47.971|
|2|48.270|47.806|48.083|
|3|48.437|47.430|47.882|

Swaps1 improves .837/.464/1.007ms/token vs2, median .837. Swaps0 improves .613/.187/.555 vs2 but is worse than1 by .224/.277/.452. This supports keeping a limited replacement budget, not deleting lookahead. Swaps0 retains prediction/claims, free-slot admission, promotion etc; NOT equivalent to --no-lookahead. Cache policy changes work/text, not same-work scheduler causality. Current prompt only; no MTP/chat/second-prompt adoption gate met.

## Independent diagnostic budget runs

Config fill-budget-diagnostic.json, /tmp/attr-fill-budget-diag-01.3restarts/arm x3requests, no drops, same lifecycle instrument. Time excluded from performance claims. MB per48-layer tick>=3 window equivalent (not aligned decode/request boundary):

|budget|demand staged|cache-fill|sum|unused-before-supersession fraction of cache-fill|
|---|---:|---:|---:|---:|
|2|188.8–190.9|168.3–169.0|357.5–359.9|37.7–38.6%|
|1|207.3–207.6|90.8–91.0|298.2–298.5|34.4–35.1%|
|0|230.4–230.9|0 in filtered window|230.4–230.9|not applicable|

2->1: cache-fill drops ~78MB while demand staging increases ~18MB, net ~60MB fewer selected H2D payload/equivalent. Unused superseded fill payload roughly halves (~64->32MB/equivalent). Diagnostic and clean are independent captures; no subtraction from old nsys capture, no payload/bandwidth-derived TPOT claim.

Crucially0 transfers less but is slower than1: byte count alone is not objective; residency/CPU placement and overlap matter. Strata's smaller selected staged volume remains policy/work different; DO NOT subtract this .837 from earlier10.657 cross-session gap.

## Next

Validate budget1 on second prompt, MTP2 and chat before default change. At fixed budget, correlate fill enqueue vs demand DMA queue/consumption to isolate exposed background-copy contention from improved placement. Instrument replace/admit reason/victim plus request ID if needed; current cache-fill classification aggregates callers. GPU/request clock bridge unresolved. No source behavior changed beyond opt-in events.

Build CUDA release passes. New lifecycle parser unit test covers used/superseded/censored distinction; full tests rerun. Failed config creation -01 had no inference capture; -02 used rebuilt binary. Analyzer initial30s timeout only parsing repeated data, rerun succeeded (no request retry).
