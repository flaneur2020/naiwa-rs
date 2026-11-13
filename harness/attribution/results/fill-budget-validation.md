# Replacement budget1: follow-up validation

Status: positive policy results in two prompts/MTP2; not default adoption. Parent: fill-budget.md / M42 S18.

All clean comparisons use Rayon/frac .4/lookahead on, budget2 vs1,3 AB/BA/AB restarted pairs x3requests/process x128completiontokens (127intervals), no diagnostic probes. CPU-first enabled except explicit baseline-order control. All runs complete without retry/failure. Text differs by corresponding request position: policy-level comparison, no same-work claim. Results/archived manifests in fill-budget-validation.json and fill-budget-*-manifest.json.

|clean condition|round1 delta|round2|round3|median paired delta ms/token|
|---|---:|---:|---:|---:|
|second prompt, no drafting|-.377|-.770|-.437|-.437|
|original prompt, MTP2|-3.080|-1.972|-.762|-1.972|
|second prompt, MTP2|-3.029|-.959|-5.332|-3.029|
|original prompt, no drafting, CPU-first OFF|-1.029|-.981|-.680|-.981|

Original no-drafting CPU-first-on result prior run -.837/-.464/-1.007 (median-.837) remains separate. Thus gain reproduced on second prompt, drafting on/off and original CPU submit order. Do not subtract medians across runs to estimate interaction or original Strata gap.

Raw runs /tmp/attr-fill-{second-01,mtp-clean-01,second-mtp-01,baseline-order-01}. Configs fill-budget-{second,mtp-clean,second-mtp,baseline-order}.json. Env CPU-before-DMA is string0 in last run; source explicitly checks ==1, so correctly off.

## Independent MTP accounting

Separate diagnostic /tmp/attr-fill-mtp-diag-01, config fill-budget-mtp-diagnostic.json.3restart pairs x3requests; terminal step records valid. See fill-budget-mtp-decomposition.json. All9 candidates lower D/N, reference89.50–92.18ms/outcome vs candidate84.71–86.97. Completed outcomes change -4..+2 (7decreases,2increases). Exact decomposition per request, not medians of components added:

- r1/request2: -5.451ms/token = count-2.763 + amortized cost-2.688.
- r3/request3: -.856 = count+1.383 + amortized cost-2.238.

All9 diagnostic TPOT pairs improve, but not a clean result. D/N includes draft/bootstrap/terminal/transport, not pure verification kernels; different histories/routes mean no causal same-work standardization. Clean runs do NOT contain counts, so diagnostic decomposition cannot be backfilled onto clean gains. Second-prompt MTP count decomposition not collected.

## Decision and remaining work

Budget1 is stronger candidate than simply disabling lookahead: replicated policy gains across two prompts and persists without CPU-first. Mechanistic support: independent lifecycle/bytes shows fewer unused superseded fills and lower total H2D despite increased demand staging. Still cannot distinguish less PCIe contention from changed CPU/GPU placement; GPU consumption/request bridge remains missing. No default changed; chat corpus, tail/TTFT regression gates and oracle/correctness acceptance for default adoption remain. Do not assume complete text equality after changed placement; audit numerical behavior if adoption requires deterministic route/output identity. No new engine code this follow-up.
