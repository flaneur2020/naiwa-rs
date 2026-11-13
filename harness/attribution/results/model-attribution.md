# Model-level narrowing and shared scheduling ablation

Status: diagnostic model brackets collected; simple shared-early experiment reverted. Parent structural-attribution.md.

Added optional fork qwen4_split boundary callback (no callback by default), model layer/mixer/MoE/routed/shared/end markers; naiwa installs only NAIWA_MODEL_PARTITION_FILE. Reusable8events/layer, numeric background output after300ms idle, drop markers, no added synchronize. Delay elapsed queries until two later layer starts after routed host reads. Last layers censored, no request IDs; single GPU serial qwen4-exp only. Shape-based rows from mixer input; no candle operator changes. Module public only to install function pointer.

Config model-partition.json, raw /tmp/attr-model-partition-01.3AB/BA/AB restart pairs x3requests x128tokens; control no probe vs model probe, diagnostic mode. Probe deltas -.019/+.717/+.500ms/token (median+.500); asynchronous residency/work changes prevent constant overhead subtraction.378complete48layer rows1groups/process, drops0; prefill/request-crossing groups structurally excluded. No exact request/position mapping. Parser model_costs.py validates endpoints/order/groups, compares separate Strata S22 raw stamp boundaries; not contemporaneous causal subtraction.

|naiwa stream bracket|range ms/48layer group|
|---|---:|
|layer start->mixer input (PLE/HC/etc)|2.978–3.030|
|mixer|7.151–7.199|
|mixer->MoE start (HC glue)|2.495–2.515|
|MoE start->routed end (router+experts)|30.218–30.326|
|routed end->shared end|1.935–1.953|
|shared end->MoE end|.236–.240|
|MoE end->layer end|.347–.351|
|layer end->next layer start (incl head/token gap)|3.165–3.687|

Strata existing GPU brackets: HC0 1.945–1.957, mixer6.385–6.399, HC1+router2.824–2.842, shared/quant .875–.881, head1.187–1.197. Naiwa router remains in30.2ms routed bracket, Strata router in HC1. Names not operator identity. Former non-MoE~6ms delta is NOT mixer-only: mixer difference is on order .8ms in separate diagnostics; HC/setup, shared, head/token progression and host gaps also contribute. Naiwa shared~1.94 vs Strata .88 not pure arithmetic difference (quant/overlap differs). Layer47 next interval contains head+sampling; cannot claim3.3ms head operator. Next needs correlated host/GPU traces and same-shape kernels, not more unconditional category subtraction.

## Single mechanism shared scheduling test

Source ordering nai​​wa shared runs after routed CPU result while Strata queues shared/quant before expert region. Tested default-off NAIWA_SHARED_EARLY, same operations and final addition but shared queued before routed experts. Current CPU-first/budget1/Rayon/frac .4/lookahead on/no drafting.3restart AB/BA/AB x3requests x128tokens on each prompt; clean no probes.

|prompt|candidate-base round deltas ms/token|paired median|
|---|---|---:|
|database|-.595/-.084/-.181|-.181|
|OS scheduling|+.125/-.484/-.673|-.484|

Text differs; policy-level only, no same-work causality. No stable~1ms improvement; does not explain original~6ms non-MoE difference. Reverted shared-early code, release rebuilt. Saved patch model-boundaries-and-shared-early.patch includes model probes too (not standalone shared patch). This negative/marginal result closes only this simple ordering implementation, NOT all shared-overlap approaches. It queues shared before CPU task submission too, potentially delaying activation/CPU work, so no general direction rejection.

Artifacts model-costs.json, model-partition/shared-early{,-second}-{manifest,report}.json; raw /tmp/attr-shared-{early,second}-01. No failures/retries. CUDA release build and release check pass after rollback; Rust58pass5ignored, Python15pass. Final source shape cleanup only affects row labeling, capture used hidden2560 equivalent; default source no shared reordering. Oracle not claimed. Defaults unchanged. Exact clean10ms allocation remains open.
