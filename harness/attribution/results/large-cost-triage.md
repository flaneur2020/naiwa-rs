# Large-cost triage against ~10ms/token remaining gap

Status: large candidates quantified on naiwa; cross-engine cause NOT closed. Parent: same-session strata-three comparison.

## Clean feature ablations

Default Rayon/frac .4/lookahead-on/CPU-first/no drafting. Three rotated ABC/BCA/CAB rounds, three128-token requests/process, each restarted. Arms current, promotion off, lookahead off; features disabled independently, not combined. Raw /tmp/attr-large-cost-01, config large-cost-ablation.json, archived manifest.

| round |current|promotion off|lookahead off|
|---|---:|---:|---:|
|1|48.118|48.964|48.671|
|2|47.892|49.144|48.441|
|3|47.820|49.065|48.109|

Disabling promotion costs+.846/+1.252/+1.246ms/token, paired median+1.246. Disabling lookahead costs+.553/+.550/+.289, median+.550. Both are beneficial policies here; do not simply delete extra transfers. Ablations change subsequent residency/routes/work and therefore cannot isolate their transfer overhead or prove optimality.

## Current nsys cost ledger

Separate diagnostic capture current arm via /tmp/nsysboth.sh; /tmp/nsys-large-cost-current.nsys-rep, /tmp/large-cost-trace.csv. Client TTFT517ms/total6811/128tokens, diagnostic TPOT49.6 is NOT a clean A/B estimate.6016 top-k publish kernels, insufficient to pretend6096 complete layer windows. Analyze last5953 publish starts =5952 layer windows, divide by124 48-layer-window equivalents. These are NOT independently verified request decode steps; request-ID/GPU clock alignment still unavailable. Keep this explicit caveat. No duration-derived suffix selection or outlier removal used.

| ledger |ms per48-layer-window equivalent|
|---|---:|
|span|50.602|
|all activity union|43.146|
|global idle|7.456|
|copy union|27.367|
|copy exclusive|17.628|
|CPU-result wait-kernel union|7.616|
|wait-kernel exclusive|4.613|
|other kernel union|17.903|

H2D347.261MB/equivalent at828.1copies; D2D62.031MB/93.3copies; D2H.684MB/139.9copies. CSV byte fields rounded. H2D duration sum27.488ms. Largest arithmetic kernel sum mmvq_gguf_q6_k4.048ms, generic gemv2.088ms, cutlass1.740ms; these are SUMS, not bottleneck or recoverable wall time. Full ledger large-cost-gpu.json.

## Interpretation

Large amounts exist in transfer exposure, result waiting and host/device idle, so focusing only on tens-of-us launches is insufficient. But these are occupancy categories, NOT additive causes of the10ms cross-engine gap, not speedup bounds. Wait kernels can overlap copies, and copy busy does not imply compute busy. Current Rayon result-wait exposure is material (~4.6ms diagnostic equivalent), unlike earlier spin configuration; historical captures must not be merged as if same config. More concurrency can change transfer/cache work as well as wait placement.

Potential transfer attribution: staged misses, lookahead admission/replacement, and promotion (primarily D2D) must be counted per source in SAME request. Old347MB-vs-staged-byte totals from different requests are not a valid residual. Current source-level clean ablations show both speculative cache policies pay back overall, not that bytes are harmless. Candidate next mechanisms are useful-byte efficiency and hiding CPU tail, not another transport substitution (already failed).

Next necessary instrument: per-request producer byte counters and stable decode boundaries, then paired timeline on/off CPU-first with identical configuration; GPU consumption correlation remains open. Matching Strata stage timer names to these sums is invalid. No default changes.
