# M43 structural refactor experiments

## R0/R1 first attempt

Status: R0 snapshot saved; R1 extraction attempted and reverted. No refactor accepted.

Snapshot `/tmp/m43-r0` contains source archives (tracked and nonignored untracked files), git patches, per-file SHA256 manifests for naiwa/fork, and copied existing executable. Archived manifest summaries here under results/. Existing binary SHA256187db5f7d153af6bb7a7a91342f7cd1fa2d5ee3dd04cbd83ae477580ca59d43a. Limitation: baseline was not freshly rebuilt before snapshot; source/binary equivalence not established, so this attempt cannot definitively attribute timing differences to extraction.

Mechanical extraction moved LookaheadChannel and five pure cache-policy functions into sibling files; original types, function bodies, atomic orderings and callers retained. CUDA build and existing Rust58pass/5ignored. Patch preserved in patches/r1-channel-policy-extraction.patch.

Clean five restarted pairs per configuration, three128-token requests/process, AB/BA/AB/BA/AB:

|configuration|candidate minus baseline per round ms/token|paired median|
|---|---|---:|
|default Rayon/no drafting, swaps2|+.464/-.734/-.832/+.447/+.320|+.320|
|physical5/MTP2 optimized configuration|-.095/+.695/+.641/-.069/-1.504|-.069|

Default exceeds planned .3ms cap (though <1%), therefore NOT accepted. Lookahead on means texts/work differ; no same-work attribution, no claim moving files inherently slows execution. Tuned MTP counts not collected. Only first prompt covered; oracle/second prompt not completed. Missing gates aren't passed by unit tests.

Raw `/tmp/m43-r1-default` and `/tmp/m43-r1-tuned`; config/manifest/report archived. Baseline arms use frozen `/tmp/m43-r0/baseline`, candidates then-current target/release. Both complete without request retry. Results are reviewable negative/inconclusive evidence, not a successful performance-neutral refactor.

Source cached_layer.rs/mod.rs restored byte-for-byte from R0 archive; extracted files removed. Restored source release build/check passes. Next attempt must first build a source-matched baseline, run oracle before changes, then use smaller extraction (pure policy only) and full five-restart gates. Do not repeatedly rerun identical candidate seeking favorable median. No defaults changed; unrelated work retained. No commits made in this task.

## R0 corrected / R1 policy-only attempt

A fresh CUDA release baseline was built before any extraction, copied to `/tmp/m43-policy-baseline/baseline`; matching source/fork archives and per-file hashes recorded in `results/policy-baseline-provenance.json`. Frozen frac0/no-lookahead/max-seqs4 oracle passed for both baseline and candidate, first line byte-identical; copies archived. Only five pure policy functions moved, bodies unchanged; activation channel stayed in place. Rust58pass/5ignored and release build succeeded.

Five-restart clean paired deltas candidate minus baseline:

|configuration|round deltas ms/token|median|
|---|---|---:|
|default Rayon/no drafting|+.050/-.130/-1.061/+.534/+.139|+.050|
|physical5/MTP2 optimized|+.591/+2.095/-.961/-3.011/+1.070|+.591|

Default passed planned TPOT cap; tuned MTP failed .3ms and1% caps. No same-work/acceptance identity was collected; text differs, so these are policy-level measurements and do not establish a code-motion cost. Do not re-run identical candidate until favorable. Candidate reverted, patch `patches/r1-policy-only.patch` and config/manifest/report archived under policy-*. No accepted refactor. Restored release build/check pass.

Combined build/test/oracle command first hit outer300s timeout after readiness, before inference; incomplete candidate oracle `/tmp/m43-policy-oracle-after` excluded. New independent oracle `/tmp/m43-policy-oracle-after-02` completed. Performance batches `/tmp/m43-policy-default-01` and `/tmp/m43-policy-tuned-01` both complete without retries. Oracle covers one16-token prompt; it does not replace two-prompt/work-identity gates, which remain uncompleted on this rejected candidate.

R0 source-matched baseline/oracle now established. Next instrumentation prerequisite: paired no-lookahead MTP outcome/work identity control, and a same-binary repeatability floor for the proposed .3ms gate. Any threshold change requires an explicit prospective plan revision; current threshold not silently relaxed. Refactor R1 still incomplete; unrelated code changes and defaults untouched.
