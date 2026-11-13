# Isolated dispatch benchmark

Run `cargo build --release --offline --manifest-path harness/m41/dispatch/Cargo.toml`.
CLI: `harness/m41/dispatch/target/release/dispatch-bench MODE PARTICIPANTS TASKS ITERATIONS GAP_US`.
Modes: rayon (real Rayon1.12), host (flat queue with N-1 workers and participating caller), flat (N workers and spinning caller). Synthetic per-task wrapping integer arithmetic, identical task IDs and outputs. No GPU/model. Final outputs checked against serial reference.50 warmup and300 measured batches; p50/p90/p99 in microseconds.

The flat queue is a Strata-inspired mechanism model, NOT the Strata ExpertPool. It waits for every worker acknowledgement on every epoch, unlike Strata's epoch-tagged claim protocol; no affinity and no native expert kernels. Workers poll an atomic epoch up to50us then sleep on a condition variable; caller waits by spinning. Rayon uses its real external install path. Six computational participants in all arms; flat has an additional non-computing caller. These differences prevent attributing the entire delta to busy polling alone.

Results in results/dispatch.csv: mode,participants,tasks,iterations,gap_us,p50_us,p90_us,p99_us. Three rounds, reversed mode order in round2; fresh process per case. Only the selected pool is constructed. GAP_US uses sleep (requested duration, not guaranteed exact). Gap0 has no intentional gap; final correctness check occurs outside measurement. Arithmetic is compute-heavy and does not reproduce expert weight bandwidth/cache behavior. No per-task timing instrumentation.

Median of three batch p50s:

|tasks|gap us|Rayon|host model|flat model|
|---|---:|---:|---:|---:|
|18|0|19.26|10.13|10.28|
|18|500|57.82|28.93|30.14|
|120|0|73.36|61.73|61.43|
|120|500|158.31|113.32|120.51|

Task work is2000 integer recurrence iterations per task.18 vs120 tasks changes total work, so these rows do NOT isolate granularity. Compare modes only within a row. Boundary latency is lower in the model, especially after sleeping gaps; host participation alone has little effect in hot batches. Not an actual Strata throughput comparison and not a TPOT savings estimate. Actual ExpertPool integration or fixed expert-input replay remains necessary for an engine-specific conclusion.

## Real Strata ExpertPool / native GGUF follow-up

`native_pool.cpp` links the existing Strata static CPU library and reads actual layer0 experts from IQ1_M GGUF. Each selected expert is copied once into a contiguous native blob; fixed IDs `(j*37+5)%256`, fixed sin activation, one row/expert. No GPU. Six computational participants: five workers plus computing caller versus six workers plus waiting caller. Explicit18 tasks/phase; affinity disabled in both arms.20 warmups,100 measured batches; output checked byte-identical within each process, finite, and checksums reported across arms (checksum equality alone is not a cross-arm bitwise proof). No weighted sum or end-to-end decode. Hot repeated weights, no inter-batch gap. Real Strata pool, not the earlier Rust model.

Build (adjust checkout paths):
```sh
g++ -O3 -std=c++17 -pthread -I../strata/include harness/m41/native_pool.cpp ../strata/build/libstrata_kernels_cpu.a ../strata/build/ggml/src/libggml-cpu.a ../strata/build/ggml/src/libggml-base.a -fopenmp -ldl -o /tmp/m41-native-pool
/tmp/m41-native-pool MODEL.gguf 0 6 5 1 18
```
Arguments: GGUF layer experts workers host_works tasks. CSV: layer,experts,workers,host_works,tasks,p50_us,p90_us,p99_us,checksum. Raw results: results/native-pool.csv. Three rounds, each host-participating arm followed by nonparticipating arm; order not counterbalanced.

| experts | host participates p50 medians across rounds | host waits | difference |
|---|---:|---:|---:|
|2|199.886 us|203.482 us|-3.596 us|
|6|598.935 us|605.206 us|-6.271 us|

Two-expert paired deltas -6.101,+34.886,-3.596 us are inconsistent. Six-expert deltas -12.513,-1.984,-37.119 us favor host participation but vary substantially. This does not demonstrate a stable large saving. It isolates host participation within real Strata at equal compute participants, not the full Rayon-versus-Strata difference. No naiwa native-kernel comparison yet; no TPOT inference. Pinning, memory bandwidth and arithmetic implementation differ from the earlier synthetic benchmark.

## Same native task bodies: actual Rayon versus actual ExpertPool

`native_rayon.cpp` uses the same native_gu_rows/native_down_rows and native_quant_h functions as ExpertPool for layer0 (IQ3_XXS/IQ4NL). Global expert-row ranges use the identical division into18 tasks per phase. Q2 specialized formats are rejected. A Rust staticlib in dispatch/src/lib.rs exposes real Rayon install/parallel-iterator through a C callback. Six compute participants: Rayon6 workers versus ExpertPool5 workers plus caller. No affinity; repeated hot weights; no intentional gap.

Every measured result is memcmp-checked against a real ExpertPool reference created before either tested pool; all passed. Three rounds, reversed arm order in round2. Native workload and partition are held fixed; scheduling, completion policy, bookkeeping and caller participation still differ. Rayon includes an FFI task callback. No GPU or naiwa fused arithmetic is used.

|experts|ExpertPool round p50 us|Rayon round p50 us|median-of-medians difference|
|---|---|---|---:|
|2|197.010 /198.883 /200.336|232.196 /294.413 /283.262|+84.379 us|
|6|711.186 /595.379 /611.439|714.733 /704.152 /642.517|+92.713 us|

All paired deltas favor ExpertPool, but six-expert round1 is nearly tied. Paired deltas are35.186/95.530/82.926us and3.547/108.773/31.078us; median paired deltas82.926 and31.078us differ from the difference of aggregate medians. Do not conflate these estimators. Raw results: results/native-rayon.csv. No engine TPOT inference. Build Rust staticlib with cargo build --release --offline, then link native_rayon.cpp with libdispatch_bench.a plus the same Strata libraries as native_pool.cpp. BENCH_RAYON=1 selects Rayon; unset selects ExpertPool.

## Three-row and inter-batch gap extension

Native CLI now requires two trailing arguments: rows (1..MAXT) and gap_us (>=0). Each expert receives all rows with distinct fixed sin activations. Scratch and output are indexed by expert and row; callbacks retain identical global output-row partitions. This is full overlap across rows, not a replay of typical sparse routing. CSV now adds rows,gap_us before p50,p90,p99,checksum. The older native-rayon.csv uses the original schema. Every iteration is memcmp-checked against actual ExpertPool reference. Verification outside timing still adds a small gap to gap0 cases. No failures in the following runs.

Three rounds, reverse arm order in round2, six compute participants,18 tasks/phase, layer0. Median paired Rayon-minus-ExpertPool p50 differences:

|experts|rows|requested gap us|ExpertPool median p50 us|Rayon median p50 us|median paired delta us|
|---|---:|---:|---:|---:|---:|
|2|3|0|275.969|310.273|34.304|
|2|3|500|271.921|675.429|400.393|
|6|3|0|831.783|890.823|55.684|
|6|3|500|820.942|1407.090|587.340|

Sleep is outside timed regions; requested500us is not a measured GPU gap. Workload is hot, repeated experts, no GPU contention. Large gaps cannot be projected to TPOT. Raw results: results/native-rayon-rows3.csv.

Counter-control with actual Strata only,2 experts/3 rows/gap500us: STRATA_POOL_SPIN_US alternates0/2000 three times. Results in native-spin-control.csv (spin is implicit by alternating rows,0 first). p50s0us:524.725,416.181,524.956;2000us:268.935,271.009,276.960. Median paired difference247.996us. Source default kSpinBeforeSleep is20ms, not the50us used by the earlier Rust model. Changing only the spin budget reproduces a substantial part of the gap under this isolated workload. It can affect wakeups, scheduling, frequency and cache state, so it does not identify a single low-level cause. No energy/runtime-occupancy measurements and no end-to-end gain claim.

## Spin-budget adoption check

Raw data results/spin-budget.csv. Actual ExpertPool,2 experts x3 rows, requested500us gap, six compute participants. Three rounds, reverse budget order in round2. All output memcmp checks passed.

|spin budget us|median of p50 us|median process CPU seconds|
|---|---:|---:|
|0|524.485|0.3779|
|50|523.474|0.3990|
|200|382.128|0.4366|
|1000|272.522|0.5739|
|2000|271.700|0.5508|

CPU time is process-wide user+system from child rusage, including startup, GGUF access, reference construction, warmup, verification and teardown. It is NOT an isolated measurement of timed batch CPU cost; cross-budget trends are preliminary. No GPU or competing token/lookahead thread.1ms retains2ms latency on this workload;50us does not.200us varies277.762-460.064us. No evidence to adopt busy polling as naiwa default. An opt-in actual-engine experiment may be justified, but requires separate correctness and diagnostic-free paired TPOT/CPU-resource validation. No engine spin implementation was added here.

## naiwa GPU fetch staging experiment (reverted)

Question: Strata's kernel-staging beat its DMA staging by ~2.3 ms/token with identical text. Does replacing naiwa's memcpy staging with a GPU read of the registered host weights help?

Implementation: new naiwa_fetch_projection kernel (uint4 when 16-byte aligned, bytewise otherwise) launched on the existing copy stream between the slot read fence and the layer's completion event. Same slot, selected experts, byte counts and ordering; only the transfer mechanism changed. Enabled by NAIWA_STAGE_GPU_FETCH. Per projection, so three launches per staged expert; host alias lookup is per projection. NOT Strata's design: no batched descriptor list, no fused fetch, arguments passed per launch.

Equality: unit test gpu_fetch_registered_bytes_match copied1,4097,65539 bytes from cuMemHostRegister'd sources at offsets0,0,3 through the fetch kernel and compared byte-for-byte on the device (clone_dtoh). Passed. This covers the kernel and alias path only, not full-model numerics.

Three alternating rounds, DMA/F,F/DMA,DMA/F, fresh engine each arm,3 consecutive128-token requests, spin layout on, frac0.6, cache2.5GiB, no MTP, no diagnostic probes.

|round|DMA TPOT|GPU fetch TPOT|fetch-DMA|
|---|---:|---:|---:|
|1|46.783|48.051|+1.267|
|2|46.217|48.075|+1.859|
|3|46.437|47.801|+1.364|

Median paired delta+1.364 ms/token, all rounds slower. Text hashes differ between arms, so the requests did not do identical work; the direction is nevertheless consistent and there is no sign of gain. Decision: revert the engine path (removed), keep the patch and result for the record under harness/m41/patches and results/gpu-fetch-ab.json.

This does not refute Strata's kernel staging: that path issues one launch for the whole group with device-side descriptors (fetch_blobs with a grid over all blobs, then rebase_ptrs), whereas this slice pays one host launch plus one alias lookup per projection per expert. The next experiment, if this line is resumed, is a single batched fetch launch per layer written from device-visible descriptor memory, not more per-projection launches.

## Batched GPU fetch with mapped descriptors (reverted)

Fixed the two defects of the first attempt: descriptors now live in one mapped pinned block the host writes directly (no per-layer memcpy, no pageable staging stall), and the kernel indexes 16-byte units instead of byte offsets (the earlier `MISALIGNED_ADDRESS`). One launch per layer instead of three per expert; device aliases cached per host pointer.

Bandwidth microbenchmark (`h2d_fetch_vs_memcpy`, registered host source, same copy stream): memcpy vs fetch kernel 13.06/12.71 GB/s at 700 KiB, 13.28/13.16 at 2 MiB, 13.38/13.34 at 8 MiB (ratio 0.97-1.00). The kernel transfers at parity; it is not a bandwidth regression.

Three alternating rounds, spin layout, frac0.6, no MTP, no probes:

|round|DMA|batched fetch|fetch-DMA|
|---|---:|---:|---:|
|1|46.30|48.07|+1.77|
|2|46.13|47.79|+1.67|
|3|46.42|47.68|+1.26|

Still slower with one launch per layer, so launch count is not the cause. Remaining explanation: the fetch kernel occupies SMs and competes with the resident expert GEMMs and the model's own kernels, while `cudaMemcpyAsync` runs on the copy engine. Reverted; patches under patches/batched-fetch-*, results in results/batched-fetch-ab.json.
