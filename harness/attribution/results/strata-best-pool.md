# Best candidate vs Strata, same-session refresh

Status: approximate parity in asymmetric drafting comparison, not stable win. Config strata-best-pool.json; raw /tmp/attr-strata-best-pool-01. Clean3rotated restart rounds ABC/BCA/CAB,3requests/process x128tokens, denominator127, no failures/retries/probes.

Candidate CPUfirst+singleGPU sampling+budget1+frac .4+physical5(5worker+caller affinity,20ms spin)+MTP2. NaiwaRayon arm identical except dispatcher/layout env absent. Strata original native/spec4/suffix0/lookup0 no drafting, all9drafts0of0 checked. Strata MTP pack missing; NOT true both-MTP comparison or equal expert work. Defaults unchanged.

|round|naiwa candidate|naiwaRayon|Strata|
|---|---:|---:|---:|
|1|37.956|39.300|37.795|
|2|36.530|38.910|37.886|
|3|38.175|37.577|37.765|

Medianprocess medians37.956/38.910/37.795. Candidate vsStrata +.161ms (+.426%); paired deltas+.161/-1.356/+.410, so not stable winner. Candidate vsRayon -1.345/-2.380/+.598 (median-1.345), third regression means prior all-round pool wins not fully reproduced. CleanMTPcounts absent; cannot explain with acceptance or execution change. Three restarts not enough to claim equivalence or stable superiority; missing StrataMTP remains capability asymmetry. No subtraction from historical no-draft10ms gap. Archived strata-best-{manifest,report}.json and draft excerpts.
