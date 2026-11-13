# Tail narrowing: LM head vs post-head progression

Status: stable diagnostic post-head interval identified; not causal cross-engine savings. Parent S25.

Default-off NAIWA_HEAD_PARTITION_FILE installs existing fork boundary callback;4reused CUDA events/token at finalHC start, LM head start/end, next model layer0start. Output buffered bounded4096channel to writer; overflow panics rather than silently dropping. No stream sync. Tensor rows+next_rows identify ordinary prefill boundaries. Final2tails censored; event elapsed queries delayed to two later tails. Host serialization/file flushing on writer, no direct token-path file writes (JSON/try_send still present). Concurrent sequences/multi-device/MTP unsupported. Layer0start is after engine/model preparation and embedding setup, not exact inference entry. head_end marks queued logits-producing ctx.lm_head, not eventual host sampling.

Initial /tmp/attr-head-01 did not record next_rows; rows1 samples included request-crossing gaps (mean4.54/2.51/2.58ms), INVALID for continuous decode interval. Fixed record next_rows, new run /tmp/attr-head-02. Three diagnostic AB/BA/AB restarted pairs,3requests x128completiontokens. Filter rows==1 and next_rows==1,377samples/process retained6excluded; request IDs absent/warmup may remain, so not claimed127perrequest. No duration outlier filtering. Parser head_costs.py validates sample minimum/finite monotonic endpoints; runner now waits for writer then validates file.

|round|finalHC ms|LM head ms|head->next model ms|probe-control TPOT delta|
|---|---:|---:|---:|---:|
|1|.0435|1.1129|1.8902|-.389|
|2|.0442|1.1148|1.8830|+.340|
|3|.0445|1.1175|1.8987|+.833|

Probe effect variable/policy changes text, median+.340ms/token; no constant overhead correction. finalHC tiny, LMhead~1.115 vs existing separate Strata GPUhead~1.19: output projection does NOT appear dominant clean gap contributor. Posthead~1.89 is stable candidate including sampling, state updates, scheduler/preparation and exposed host gaps. Cannot call it sampling alone or all removable. Strata corresponding endhead->next model boundary not measured, so no comparative1.89ms speedup claim. Old~3.2ms tail bracket is now roughly head1.1+posthead1.9 plus setup; diagnostic windows differ, no exact subtraction.

Source sampler.rs has both device greedy path and host-vector fallback. sample() checks can_sample_greedy_on_device, sparse penalties/bias, then sample_greedy_on_device; fallback paths may read full logits. Have NOT measured active path, do not assume fallback or CPU argmax. Next tag actual sampler branch and sample begin/return plus engine step boundaries using shared request/step identity. No behavior/default changes. Raw head-costs.json/head-overhead.json. Release build/check, Rust58pass5ignored/Python15pass.
