# Same-session three-arm no-drafting comparison

Status: completed. Parent: M42 / current Strata gap refresh.

Config strata-three-arm.json; run /tmp/attr-strata-three-01. Three rounds rotated ABC/BCA/CAB: each process restarted, three128-completion-token requests/process, one identical database prompt, seed0/temp0, streamed usage128 confirmed, denominator127. No failures/retries. Runner process-group cleanup; unique Strata engine config/log per arm. Strata's nine requests explicitly report drafts accepted0of0 (three draft-check files archived). No diagnostic probes enabled.

naiwa: Rayon default, frac .4, cache2.5GiB, lookahead on, deterministic combine, no MTP. CPU-first differs only NAIWA_CPU_BEFORE_DMA=1. Strata: existing expert-cache800 config, native IQ, spec4 required for native execution, suffix-draft0/lookup-chain0, no MTP. This is practical configured-engine comparison, not identical cache capacity/workload/operator boundary. Text/route identity across engines not claimed. Public SSE TTFT/decode boundary, not internal Strata generate timer. Prefix reuse histories differ between engines; compare same request positions or per-process medians, not raw timer subtraction.

|round|naiwa base|CPU-first|Strata|CPU-first minus base|CPU-first minus Strata|
|---|---:|---:|---:|---:|---:|
|1|48.485|47.407|37.136|-1.077|+10.272|
|2|48.320|47.792|37.152|-.528|+10.640|
|3|48.683|48.153|37.127|-.530|+11.026|

Median process medians: base48.485, CPU-first47.792, Strata37.136ms/token. Difference of medians10.657ms/token (+28.697%); median paired gap10.640. Median paired internal CPU-first improvement.530ms/token (not the .692 difference of medians). Distinguish estimators explicitly.

Interpretation: CPU-first wins all3 current default-Rayon/no-drafting rounds; substantial gap remains. Historical45.904vs37.047 used spin/frac .6 rather than current Rayon/frac .4 and another session: DO NOT interpret updated larger gap as regression or subtract old/new absolute times. New CPU-first remains opt-in pending lookahead/MTP/chat consistency and critical-path attribution. No claim of matching MTP capability.

Evidence: strata-three-{manifest,summary}.json and strata-three-r*-draft-check.txt. Runner now supports two-arm AB/BA or three-arm rotated order, {run_dir} config placeholders and owned Strata exec adapter. Template includes all native/spec/cache settings. GPU-consumption analysis still open.
