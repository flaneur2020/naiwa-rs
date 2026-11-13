# Five-restart comparison: neural MTP vs suffix drafting

Status: complete5round new run, not appended to old3rounds. Config strata-suffix-comparison-5round.json; raw /tmp/attr-strata-suffix5-01. Three arms rotated ABC/BCA/CAB/ABC/BCA, each5fresh processes x3requests x128completiontokens,45requests total; denominator127, no probes/failures/retries. Five-round rotation not fully order-balanced (6would balance); no independent-request sample inflation.

Naiwa best physical5(5worker+caller20ms spin/affinity),CPUfirst,GPU sampling,budget1,frac .4,lookahead on,MTP2. Strata original binary spec4, suffix3 vs suffix0, lookup-chain0. Suffix3 means minimum suffix match length, NOT3neural draft depth. Strata suffixlookup not trueMTP; native neural pack missing. Arms drafting metadata off on Strata denotes no nai​​wa MTP CLI, explicit drafting_note/comparison_note and actual logs distinguish active suffix lookup. No identical expert/text work claim.

|round|naiwa MTP2|Strata suffix0|Strata suffix3|
|---|---:|---:|---:|
|1|36.805|37.849|37.740|
|2|37.784|37.797|37.791|
|3|36.702|37.841|37.796|
|4|36.318|37.827|37.853|
|5|38.220|37.913|37.794|
|median|36.805|37.841|37.794|

Difference medians candidate vs suffix3 -.990ms/token (-2.619%). Paired improvement .935/.006/1.094/1.535/-.426, median .935; wins4of5, round2 near tie. Naiwa range36.318–38.220 vsSuffix37.740–37.853, more variable. No equivalence/superiority statistics from5samples. No MTP acceptance collection in clean; can't explain variability as acceptance vs cost.

Actual suffix3 eachrestart: firstrequest0of0; second1of2; third1of1. Total5restarts suffix10accepted/15proposed, only2suffix windows/process. Tiny actual draft coverage, not neuralMTP equivalent; matches weak timing gain paired median .045ms/token vsSuffix0. Do not imply Strata fullMTP was benchmarked. Fivefresh restarts all native draft logs archived. suffix0all0of0. Full45request provenance/report saved strata-suffix5-{manifest,report}.json; log excerpt strata-suffix5-drafts.txt.

Defaults unchanged. Report is per-process medians over3requests, then median across5restarts; don't pool15requests as15restarts. git diffcheck passes. No engine source change this run.
