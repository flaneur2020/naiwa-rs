# Refreshed clean comparison: CPU-first + replacement budget1

Status: complete. Parent: fill-budget-validation / M42. Config strata-budget1-three-arm.json; raw /tmp/attr-strata-budget1-01. Three3-arm rotated ABC/BCA/CAB restart rounds, three128-completion-token requests/process, denominator127, no probes/failures/retries. Same database prompt, greedy. Base default Rayon/frac .4/lookahead on/budget2/CPU-first off; candidate differs CPU-first on AND budget1. Strata original engine binary/expert-cache800 config, native spec4 with suffix0/lookup0; all9requests drafts accepted0of0 (archived). Not patched ledger build. Practical configured-engine comparison, cache capacity/internal routing/text not aligned. No MTP comparison implied.

|round|naiwa base|CPU-first+budget1|Strata|candidate-base|candidate-Strata|
|---|---:|---:|---:|---:|---:|
|1|48.589|47.621|37.864|-.968|+9.757|
|2|49.248|47.635|37.903|-1.613|+9.732|
|3|49.013|48.312|38.001|-.701|+10.311|

Median process medians49.013/47.635/37.903ms/token. Difference of candidate-vs-Strata medians9.732ms (+25.676%); median paired gap9.757. Median paired candidate improvement over base .968ms/token. Difference of base/candidate medians1.378ms is a different estimator; do not call it paired gain. Combined change not per-mechanism attribution. Prior refresh had different absolute times; no cross-session subtraction of gap to claim incremental improvement.

Current conclusion: candidate wins all3rounds but remaining~9.7–10.3ms gap still substantial. Defaults unchanged; request/compute clock bridge/chat adoption unfinished. Evidence strata-budget1-manifest.json, report.json and r*-draft-check.txt. git diff check passes.
