# CPU pool tuning on current optimized combination

Status: single-prompt positive physical5 results, unbound spin8 regression; opt-in only. Parent current tuning validation.

Current CPU-first/budget1/SINGLE_GPU_SAMPLE/Rayon baseline, frac .4/lookahead on. Compare default Rayon vs 8worker+caller20ms spin unbound vs5worker+caller20ms spin with physical-layout affinity. Physical layout binds workersCPU1–5/callerCPU0, six physical cores, not just threads setting. Joint dispatcher+count+affinity intervention; cannot isolate any one contributor. Three rotated ABC/BCA/CAB restarts, three128completiontoken requests/process, denominator127, no diagnostic probes. MTP fixed2 in separate session from no-drafting; don't subtract them as drafting ablation. Text/work identity not established.

|mode/round|Rayon|spin8|physical5|
|---|---:|---:|---:|
|MTP2/1|39.442|50.103|36.739|
|MTP2/2|38.104|49.249|36.392|
|MTP2/3|40.633|51.882|36.779|
|MTP2 median|39.442|50.103|36.739|
|no draft/1|45.918|56.145|42.967|
|no draft/2|46.198|54.264|42.537|
|no draft/3|45.973|56.284|41.911|
|no draft median|45.973|56.145|42.537|

Physical5 paired gains MTP2 2.704/1.712/3.854ms (median2.704), no drafting2.951/3.661/4.062 (median3.661). spin8 regresses~10–11ms; don't blindly enable spin without appropriate thread/layout. MTP acceptance not traced in clean, so execution-vs-count decomposition unavailable. Spin burns CPU and contends with other work; contention explanation hypothesis, no scheduler capture.

TTFT physical5 vsRayon improves all3MTP rounds (~19/18/36ms); no drafting improves4/21/6ms. SSEp99 (not decode/verify tails) no draftingRayon61.6/62.1/63.1 ->54.1/57.5/56.4ms; MTP106.7/112.0/104.7 ->99.1/98.5/105.5 (thirdslightlyregresses). Perprocess median requestp99, not pooled restart samples. Default unchanged; secondprompt/chat, deterministic-control/workidentity and acceptance/CPUwaitdiagnostics needed. No new same-session Strata comparison; cannot say36.739wins oldStrata37.771.

Configs pool-tuning-{mtp2,nospec}.json. Raw /tmp/attr-pool-mtp2-01 and /tmp/attr-pool-nospec-02. Initial combined tool900s timeout interrupted no-draft-01 after5arms, marked failed/excluded; owned remaining processgroup3377651 terminated, new full-run-02 independent, no inference retry within valid runs. Valid runs all requests complete. Archived pool-*-manifest/report.json. Engine source/defaults unchanged; Python15tests and diffcheck pass.
