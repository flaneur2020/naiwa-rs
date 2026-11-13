# MTP-enabled naiwa vs available Strata configuration

Status: asymmetric capability comparison only. Both-engine true MTP comparison blocked by absent Strata draft pack (dense.txt/experts.bin etc); available mtp GGUF not equivalent to Strata runtime pack. Main model pack dense.bin is not draft artifact.

Config strata-mtp-asymmetric.json, raw /tmp/attr-strata-mtp-03. Three rotated ABC/BCA/CAB restarts x3requests/process x128completiontokens, denominator127, no diagnostic probes/failures/retries. Naiwa CPU-first/budget1/Rayon/frac .4/lookahead on/greedy/explicit MTP2; candidate additionally SINGLE_GPU_SAMPLE. Strata original binary native/spec4/suffix0/lookup0, all9logs drafts0of0 archived. Global drafting=on with per-arm override off for Strata; runner validates explicit depth per on-arm. Two config validation failures before launch (-01 descriptive drafting string, -02 missing per-arm override) excluded, no inference retries.

|round|naiwa MTP2|MTP2+GPU sampling|Strata no drafting|
|---|---:|---:|---:|
|1|37.457|37.628|37.681|
|2|39.325|39.251|37.869|
|3|39.046|39.032|37.771|

Median process medians39.046/39.032/37.771ms/token. Latest candidate difference1.262ms (+3.340%); individual gap-.053/+1.382/+1.262. This does NOT mean equal MTP engines or parity of per-step execution. SingleGPU sampling marginal/no stable MTP benefit (delta+.172/-.074/-.014, median-.014); no-drafting sampling gain not automatically applicable to speculative driver. Clean counts not captured, acceptance/timing variation not attributed to kernel speed. Main clean off/off~10ms historical gap remains a different workload; cannot subtract to claim measured MTP benefit across sessions.

Need Strata valid MTP pack then equivalent-depth/greedy run and per-request counts; current full true-MTP comparison blocked. Existing GPU sampling remains opt-in, no new defaults. Python15tests pass, diff check passes. Evidence manifest/report plus r*-drafts.txt.
