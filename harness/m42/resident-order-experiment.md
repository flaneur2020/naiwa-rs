# M42 resident-first experiment (withdrawn)

Reproduction: `resident-order-ab.sh` requires the withdrawn experimental implementation and the local `/tmp/m41-nospec-naiwa.sh` runner. It is an evidence script, not an active supported benchmark.

Minimal implementation in `src/moe/decode.rs`:
- After Route::submit and capturing resident_prefix, if NAIWA_RESIDENT_FIRST=1 and staging enabled, wait_preload, call Gpu::submit on plan.hits, and record all resident read fences.
- Then run unchanged DmaMiss::submit and CpuMiss::submit (synchronous activation; async disabled).
- At original resident submit location, reuse saved (Tensor, Bank) instead of submitting twice.
- Preserve resident + staged + CPU addition ordering.
- On any fence error synchronize device as on baseline.

Potential hazard evaluated: blocking activation D2H on the compute stream can now wait for the early resident work, delaying CPU job start. This is a source-level explanation, NOT measured causal attribution; no event timeline was captured for this experiment.

Build succeeded; tests 58 passed, 5 ignored. frac=0/no-lookahead oracle first line byte-identical. CLI preflight supplying --mtp-n-predict without --mtp failed before model load; no performance request retried. No-drafting experiment omits that CLI option because clap requires --mtp when supplied.

Three alternating restart rounds at frac=.25; physical layout, five workers+caller, 20ms spin, deterministic combine enabled on both arms. Three requests/process, 128 completion tokens/request. TPOT=(total-first_content)/(completion_tokens-1). Results include full text SHA256.

baseline medians: 44.977 / 45.688 / 45.631 ms/token
resident-first: 46.967 / 47.763 / 46.899 ms/token
paired differences: +1.990 / +2.076 / +1.269 ms/token

IMPORTANT: hashes vary within both arms. Thus source-level partition rule/operator/order are held constant, but actual routed experts and bytes are not proven identical across requests. These are exploratory end-to-end outcomes, NOT a strict same-work trace comparison. Does not reject expert overlap as a direction. Implementation reverted; release binary rebuilt after revert.
