# T2–T4: CPU submission before staging DMA

Status: positive on repeatable no-lookahead control; retained DEFAULT OFF (`NAIWA_CPU_BEFORE_DMA=1`). Production/default adoption NOT approved.
Parent: attribution/TODO.md and M42.

## Mechanism

`DmaMiss` chooses the identical staged set under the existing ring mutex, then invokes a hook BEFORE DMA enqueue. The hook filters that set out of CPU misses (preserving order), performs the unchanged synchronous activation acquisition, and submits the remaining CPU job. Original staging, fences, assignment updates and final sum remain unchanged. Fallback when no ring/copy support uses original CPU submission. Baseline executes a no-op hook; one extra dynamic no-op call is a small instrumentation difference. CPU first may delay staging enqueue; this is an intentional scheduling tradeoff, not free overlap.

No async CPU activation, no lookahead, frac .25, five workers+caller physical layout, spin 20ms, deterministic combine, no drafting. Explicit MTP-depth omission required by CLI in no-drafting mode.

## Gates and evidence

- Build and Rust tests pass: 58 passed/5 ignored.
- frac0/no-lookahead first oracle line byte identical: /tmp/todo-cpufirst-oracle.
- One-request diagnostic: three pairs, 6192 records/pair match completely.
- Three-request diagnostic: three pairs, 18480 records/pair match completely. Exact IDs, f32 weights, residency and final assignment; not proof of activation or physical byte identity.
- Work records are perturbing diagnostics, not performance measurements.
- Raw diagnostics /tmp/attr-cpufirst-diag-01 and /tmp/attr-cpufirst-diag-03req; summaries cpu-first-diagnostic*.json.

## Clean AB / BA / AB

Three restarted rounds, three requests/process, 128 completion tokens each; TPOT denominator 127. /tmp/attr-cpufirst-clean-01; archived manifest cpu-first-clean-manifest.json.

| round | base median ms/token | CPU-first | paired delta |
|---|---:|---:|---:|
| 1 | 43.177 | 42.270 | -0.907 |
| 2 | 43.508 | 42.289 | -1.219 |
| 3 | 43.439 | 42.603 | -0.835 |

Median paired improvement .907 ms/token. Each corresponding request position improves in all three rounds. SSE content-gap p95 improves 54.630→52.888 / 54.413→53.059 / 56.042→53.444 ms; p99 61.798→60.328 / 62.263→60.906 / 62.876→60.689 ms. These are SSE gaps, NOT engine steps, and per-process pooled across three requests.

Global all-text-equal is false because first-request text differs from later-request text. This is NOT an arm divergence: all six first requests have SHA256 5d35365870b2f2495e87d5471eca60ed620b8fa7ddd5bbc0bacd94f6611777aa; all twelve later requests have SHA256 4a54db4d18782755a166937bf5cd2fae53c4acfb98afcfc6be59206ff678ef23. Request history/reuse changes the workload; compare corresponding positions. Harness now reports equality by position explicitly.

## Limits / next checks

Observed scheduling-policy benefit on this control, not a measured CPU/GPU critical-path decomposition. No GPU/CPU correlated timestamp capture yet. Need second prompt and production lookahead-on checks, including tails and demand differences, before default adoption. Do not bundle mandatory activation-channel changes with this implementation. Hook currently submits before ring acquire, under ring lock: CPU work does not acquire that lock, but audit future lock dependencies when changing executor/slot handling.
