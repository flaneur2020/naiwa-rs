# Single-sequence greedy GPU sampling: fallback attribution

Status: opt-in mechanism with two-prompt clean wins, repeatable-text negative control. Defaults unchanged. Parent S27 sampler-narrowing.md.

## Actual rejection found

pipeline/sampling.rs try_sample_batch_cuda rejects seqs.len()==1 && !categorical AFTER successful recognizer/tool/plan checks. sample_and_add_toks_inner then into_cpu_rows -> sample_sequence -> CPU Sampler::sample. Instrumented each reject reason and CPU conversion duration. Initial /tmp/attr-pipeline-sample-01 before final single reject marker had no plan/tool reject; CPU conversion cumulative554–564ms/process and sample host487–489ms/process, but windows include warmup and GPU synchronization, not pure costs.

Opt-in NAIWA_SINGLE_GPU_SAMPLE=1 removes ONLY this single-seq rejection for plan.kind.is_argmax() and !sampling_logprob_required. Existing checks remain (recognizer/tool, penalties/DRY/bias/processors/logprobs, device/shape). Existing sample_cuda_top1_batch + sample_cuda_top1_row reused, no new kernel, no stochastic topK change. Mixed categorical guard unchanged. Default off, no semantic bypass of unsupported cases. Names use generic argmax plan; topK1 may also qualify per existing is_argmax, not temperature-only check.

## Clean gains

Current CPU-first/budget1/Rayon/frac .4/lookahead on/no drafting.3AB/BA/AB restart pairs x3requests/process x128completiontokens, denominator127, no probes/failures/retries.

|prompt/control|round1 delta|round2|round3|median paired ms/token|
|---|---:|---:|---:|---:|
|database|-1.743|-.907|-1.811|-1.743|
|OS scheduling|-1.681|-1.887|-2.221|-1.887|
|database, both no-lookahead|-2.824|-2.403|-1.969|-2.403|

Lookahead-on text differs (asynchronous policy comparison); no-lookahead ALL corresponding request positions match full text across all6processes. Global all_text_equal false from first-vs-reuse content differences. No same-route-work proof yet, no universal logits parity/tie/NaN guarantee. Negative control strengthens correctness confidence but does not replace oracle, MTP/chat and corner-case tests.

Raw /tmp/attr-single-gpu-{01,second-01,nolook-01}, configs single-gpu-sample-{clean,second,nolook}.json, archived single-gpu-*-manifest/report.json. Do not subtract new gains from old Strata9.73ms gap; new same-session refresh required.

## Independent diagnostic closure

/tmp/attr-single-gpu-diag-01,3restart pairs x3requests, same config plus head sampler probes. Baseline every process384pipeline_reject_single_or_mixed,384Sampler::sample calls; opt-in0both (GPUbatch fast path bypasses Sampler::sample, so0does NOT mean no sampling). Not instrumented success count directly; source and reject/call disappearance plus output confirm path, no direct cuda API trace claimed.

|tail|baseline|opt-in|
|---|---:|---:|
|LMhead ms|1.113–1.117|1.114–1.115|
|head->nextmodel ms|1.878–1.911|.543–.567|

Thus this control-flow choice accounts for~1.3ms of exposed posthead span IN DIAGNOSTIC capture. Clean total gain1–2ms can additionally reflect policy/overlap changes; don't force equality. Host sampler and CPUcopy durations overlap GPU waiting and must NOT be added to these spans. This is concrete narrow causal scheduling/placement intervention, not exact attribution of entire10ms.

## Gates

Still default off. Need oracle/tie/finite behavior parity, secondprompt no-lookahead identity, MTP2 and multi-turn chat, penalties/tool/logprob fallback tests. Current checks exclude these unsupported paths but tests not yet added this turn. CUDA release build/check and Rust58pass5ignored/Python15pass. Source patch single-gpu-sampling-and-probes.patch includes diagnostic additions, not minimal standalone feature patch. Candle operators unchanged.
