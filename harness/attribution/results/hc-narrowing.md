# HC narrowing: mix path, not generic missing fusion

Status: diagnostic3restart controlled partition; no behavior optimization. Parent model-attribution.md.

Expanded model callback boundaries after attn norm/mix, after mixer->mlp combine/norm and mlp mix.12reused events/layer vs8prior, numeric idle writer; no candle operator edits. Captures incompatible endpoint schemas; old JSON retained, current parser expects12phases. Config model-partition.json; raw /tmp/attr-hc-partition-02,3AB/BA/AB restart pairs x3requests/process x128tokens. Each probe process378complete decode48layer groups, no drops. Report hc-model-costs.json, overhead hc-overhead-report.json. Diagnostic only; actual request/position IDs unavailable.

Failed -01 rejected before idle writer flush; runner now sleeps2s before GPU/model artifact validation as already done for schedule. Failed run retained/excluded; no request retry. File later contains complete data, but not retroactively relabeled valid. -02 fresh all3rounds passes. Need future explicit terminal writer protocol for production assurance; fixed sleep has failure-kill limitations.

Mean3restart nai​​wa stream intervals ms/48layer group:

|interval|ms|
|---|---:|
|layer start -> attn norm ready|.740|
|attn mix|2.197|
|attn inject / gap -> mixer|.236|
|mixer|7.182|
|mixer end -> mlp norm ready (combine/norm)|.364|
|mlp mix|2.109|
|mlp inject / gap -> MoE|.234|
|router+routed experts|30.114|
|shared|1.949|
|MoE addition|.239|
|layer tail combine|.349|
|next layer/token gap|3.172|

Attn norm bracket includes PLE/setup; mlp norm bracket includes preceding attn combine. Not isolated norms. HC mix sum4.306ms is main component within earlier5.5ms setup, not norm/combine itself.

Strata existing33stamps/layer raw from /tmp/attr-engine-partition-02: first HC read0 splits at indices0/27/28/1. Per48layers norm .331–.332, down .807–.815, up/post .804–.811, total~1.95. Compare nai​​wa attn mix~2.20 vs Strata down+up/post~1.62 as descriptive bracket difference~.58ms; different normalization/dtype/layout/graph context, no clean speedup or same-op claim. Second half HC+router not identically split; don't double .58 to claim1.16ms benefit.

Source review matters: nai​​wa already has CUDA fused hc_norm/hc_mix/hc_combine; not generic unfused RMSNorm. Remaining mix is down.forward -> division -> SiLU -> up.forward -> fused gated reduce. Strata fused_gr_read_multi has specialized kernels. Therefore target low-rank projections and intermediate activation path under equal shapes, rather than replacing existing fused norm/combine or broad 'framework overhead' label. Inject brackets together~.47ms; not primary candidate. Whole mixer~.8ms diagnostic difference remains too small to explain original~6ms coarse gap.

Next exact attribution requires projection-level markers or launch-correlated kernel examples with same dtype/weights and graph context. No new behavior/default changed. Release CUDA build and cargo check pass, Python15pass. Prior Rust58/5ignored not rerun after marker-only extension; no fresh Rust-test claim.
