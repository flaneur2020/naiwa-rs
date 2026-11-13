# Structural attribution: closed diagnostic partitions, bounded causal claims

Status: both engine stream ledgers collected and closed; exact causal allocation of clean gap NOT established. Parent M42 S21. Defaults unchanged.

## Method and perturbation gates

Added opt-in NAIWA_GPU_PARTITION_FILE:9 CUDA events/layer from post-host-route through submit/preload/resident/staging/staged/promotion/CPU upload/combine, plus next route event. Record elapsed from layer route; two later host routing reads ensure completed work. Events reused via3entry pool; bounded numeric buffer65536rows; writer drains after300ms idle, flush/drop records. No new stream synchronization. Unsupported multi-device/concurrent thread association; last2layer tail censored, no forced synchronization. CPU-only path unchanged when disabled (unit-test regression initially found unconditional CUDA lookup, fixed gated optional stream). Allocation/query calls still perturb host; event placement itself shifts policy/routing.

Initial v1 created events/JSON/file writes hot path and added paired median5.920ms/token in /tmp/attr-partition-overhead-01: NOT valid for clean-gap allocation. v2 reuse/background writer overhead control /tmp/attr-partition-overhead-02,3rotated restarts x3requests:

|round|no probes|host only|host+GPU events|
|---|---:|---:|---:|
|1|48.114|47.213|48.442|
|2|47.976|47.594|48.946|
|3|47.731|47.994|49.372|

Probe-vs-no-probe deltas+.328/+.969/+1.641ms, median+.969. Host-only deltas-.901/-.383/+.263: asynchronous cache policy changes make perturbation non-additive. Do NOT subtract .969 uniformly or from one bucket. Overhead control mode diagnostic (even arm named clean), not new performance acceptance.

v2 cross-engine raw /tmp/attr-engine-partition-02,3 AB/BA/AB restart pairs x3requests x128completiontokens. Naiwa CPU-first/budget1/default Rayon/frac .4/lookahead on, Strata patched verifier/no drafting. Nine request client timing medians/process medians48.541vs38.668; diagnostic gap9.873. Not interchangeable with clean47.635vs37.903 (+9.732) from earlier refresh. Strata GPU profiler changes shared-stream scheduling; raw GPU range omits outer run/commit/emit. Source patch as S21, no Strata timing code changed this turn.

Parser validates phases/order/no drops and selects complete48layer rows1 sequences followed by next rows1layer0 (378groups/process). Excludes structurally incomplete/prefill/request-crossing groups, not duration outliers. No request IDs on naiwa; cannot claim all127service intervals/request. Strata128T1positions x48layers/request validated. Rolling nai​​wa route->next-route includes head at layer47, whereas Strata starts before layer0 dense and ends afterhead. Useful coarse analytical alignment, not identical operators/history.

## Closed coarse stream ledgers (mean3processes)

|coarse bucket|naiwa ms/48layer group|Strata ms/T1window|diagnostic delta|
|---|---:|---:|---:|
|non-MoE + progression|19.627|13.283|+6.344|
|post-route plan/submission + preload dependency|3.653|.277|+3.376|
|resident bracket|4.001|1.539|+2.463|
|staging wait bracket|10.488|11.352|-.864|
|staged bracket|1.560|.521|+1.040|
|CPU result ready bracket|8.898|8.277|+.620|
|promotion/combine tail|.712|.382|+.329|
|SUM|48.938|35.631|+13.307|

Full arithmetic in engine-partition-aligned.json; raw per-round phase means engine-partition-comparison-v2.json. Tables close separately by telescoping interval endpoints; bucket deltas sum13.307, NOT clean9.732. This mismatch is explicit, not allocated away: different endpoint sets, sampled groups, GPU-vs-client clocks, host gaps, prompt/sample timing and instrumentation, plus separate runs for clean. Strata client~38.668 vs GPU~35.631 demonstrates uncovered outer portion; do not distribute residual without boundaries.

### Bucket semantics (critical)

- Naiwa non-MoE: previous combine end -> next MoE post-host-route. Contains dense/shared/attention/router plus host route read/classification, framework submissions, head/sample between tokens and probe collection. Strata non-MoE is dense/router/shared/head/gaps stream stamps. So+6.344 does NOT establish dense arithmetic6ms slower. It identifies dense/framework/progression as largest unresolved bucket.
- Naiwa plan: event post-host-route -> CPU submission completed event; includes idle awaiting host, input D2H and DMA submission. Strata plan: shared end -> resident begin includes plan flag/copy; shared alignment differs. About+3.376 is hypothesis for dynamic host planning/dependency, not pure dispatch.
- Resident/staged brackets include host enqueue gaps/fences/reductions; naiwa has different resident/staged counts and quant formats. Higher bracket time not proof kernel is2–3x slower. Need equal expert/quant/shape microbench or launch-correlated timeline.
- Staging: naiwa event wait, Strata flag wait+fetch+rebase. Naiwa does NOT show a larger staging bracket in this capture even with more payload. Extra bytes cannot by themselves explain clean10ms; prior simple PCIe-centric interpretation incomplete.
- CPU ready: naiwa includes result wait+mapped copy, Strata wait bracket excludes copy. Difference small compared to overall diagnostic gap; default-Ray​​on CPU tail alone is not supported as dominant explanation here.
- Host ledgers overlap all above and are never added. Strata pool~22ms vs naiwa worker~25ms not same operator/work boundary.

## What is causally proven vs still descriptive

Proven policy effects in separate clean paired runs: CPU-before-DMA improves repeated no-drafting controls; replacement2->1 improves two prompts/no-drafting/MTP2 and CPU-first off. Independent identity shows36–38% unconsumed-before-supersession cache-fill, budget1 reduces net payload~60MB/equivalent and unused payload~half. Does not establish causal per-byte speedup or additivity with CPU-first.

Remaining10ms cannot honestly be numerically assigned yet. Ranked NEXT experiments:
1. Split non-MoE bucket with actual model layer start/router/shared/head/step-complete markers and request/position IDs in fork; current MoE-only hook cannot isolate it. Compare launch-correlated GPU arithmetic and exposed host idle under same boundaries. Do not change candle operators.
2. Split route->resident into activation acquisition, host plan/DMA enqueue and preload wait GPU intervals with common IDs; v2 suggests substantial dynamic-host side.
3. Equal selected expert/quant/shape grouped arithmetic benchmark against Strata; compare resident/staged counts and GPU launches, not wall brackets alone.
4. Device/request clocks and outer verify/commit/emit bridge to close client-vs-GPU residual on BOTH engines. No automatic extrapolation from377/378structural groups.

Decision: original focus on background-fill byte efficiency was useful and yielded budget1, but new complete MoE/non-MoE ledger shifts priority to non-MoE/framework progression and dynamic submission. No evidence for removing preload correctness waits, freezing caches, adopting GPU fetch or declaring CPU operator slowdown. Full descriptive structural comparison is now available; full causal10ms decomposition is explicitly NOT done.

Validation: CUDA release build succeeds; Python15tests pass; Rust58pass5ignored after CPU-path gating fix. HIP/SYCL untested. New probes default off, final gating fix source newer than capture binary; only removes disabled-path device lookup, no armed CUDA probe change. No oracle correctness gate claimed. Artifact/provenance limitations remain as README.
