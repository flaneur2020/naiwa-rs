# Projection-level narrowing and probe density control

Status: narrows HC difference to modest per-layer brackets; not major-gap causal allocation. Parent hc-narrowing.md.

Added optional model callback marks inside GatedResidual::mix after down, /hc+SiLU, up; existing marker after fused gated reduction. No operator changes. Final head HC mix ignored by layer recorder after layer_end.18events/layer; sparse stride env samples layer%stride==0 (all selected layers0/8/16/24/32/40 are GDN, NOT representative QSA sample). Native callback default absent, no probe allocations when disabled.

Dense projection capture /tmp/attr-hc-proj-01,3restart pairs x3requests x128tokens; validated no drops/full groups. TPOT perturbation+2.429/+3.026/+1.552ms, too high for numerical clean-gap interpretation. Dense firsthalf per48layers down~.90ms, act~.354, up~1.015, reduce~.229. Do NOT assign act .354ms as removable overhead, event placement itself perturbs.

Sparse capture /tmp/attr-hc-sparse-01, config hc-projection-sparse.json, same3restart pairs and requests, stride8. TPOT deltas+.236/+.235/+.006ms/token vs no-probe arm, median+.235. Reduced aggregate overhead does not eliminate local event distortion. Sparse parser validates no drops/phases/monotonicity; no actual request IDs, rows1 includes model warmup. Therefore do not extrapolate sampled layer totals to all48 or claim startup-free per-request timing. next_layer now spans8layers (48->0 at wrap); excluded from sparse sums. Current parser accepts18phase schema only; previous raw schema requires historical parser version.

## Sampled first-half HC mix, microseconds/layer

|layer|naiwa down|naiwa act|naiwa up|naiwa reduce|Strata down|Strata up/post|
|---|---:|---:|---:|---:|---:|---:|
|0|26.0–26.6|6.5–7.1|34.7–36.2|3.7–3.9|16.1|16.5|
|8|17.9–18.1|6.9–7.4|19.9–20.3|4.3–4.8|16.6|16.6|
|16|20.5–20.8|8.7–9.4|22.1–23.6|4.7–5.0|16.3|16.6|
|24|19.3–19.4|8.4–9.2|20.8–21.0|4.6–4.8|16.6|17.5|
|32|19.9–20.4|8.9–9.1|21.5–21.8|4.7–4.9|16.0|17.0|
|40|19.0–19.5|8.4–8.9|21.0–21.5|4.5–4.7|16.0|17.7|

Strata from separate existing /tmp/attr-engine-partition-02 captures positions indexed27/28/1; diagnostic contexts and dtype/implementation differ. This is matched layer index, NOT identical inputs/weights/operator boundary. Strata up/post likely incorporates operations separate in nai​​wa; do not compare up-only as equivalent. Current native kernel source specialized fused_gr vs nai​​wa generic down + activation + up + reduce path; supports bounded candidate, not full10ms explanation.

Secondhalf Strata indices30/31/17 last bracket includes router: measured~39us is NOT HC up alone. Exported raw matched-layer values hc-strata-matched-layers.json intentionally retained but don't use to claim secondhalf up slower. Need another stamp before router for same boundary if following this direction.

Conclusion: low-rank projections/intermediate launches differ by modest per-layer amounts on sampled GDN layers; sparse data do not support HC projection as many-ms dominant cause. Layer0 anomaly could include warmup/request-boundary/probe behavior and has no request alignment; do not label kernel pathology. Next priority moves back to host route/planning gap and true head/sample boundary; no new fusion experiment justified yet. Detailed raw hc-projection-costs.json, hc-projection-sparse.json, hc-projection-overhead.json, hc-strata-matched-layers.json. Defaults unchanged. Release build/check, Rust58pass5ignored and Python15pass.
