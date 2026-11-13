# T4b CPU-first follow-up

**Status:** verified second prompt; lookahead-on policy comparison positive but workload-divergent. Retain DEFAULT OFF. **Parent:** cpu-first.md / attribution/TODO.md.

## Fixed settings

Same executable as T2–T4; frac .25, physical five workers+caller, spin 20ms, deterministic combine, no drafting, synchronous activation. Each clean experiment has AB/BA/AB restart rounds and three 128-completion-token requests/process. Actual usage denominator 127. No request failures or retries. No cross-session absolute time comparison. These are experimental settings, not default production (Rayon/frac .4) or MTP/chat acceptance.

## Second prompt: OS thread scheduling, no-lookahead

Config cpu-first-second-prompt.json; raw /tmp/attr-cpufirst-second-01. Full clean manifest and per-round SSE tails archived as cpu-first-second-{manifest,tails}.json.

| round | baseline ms/token | CPU-first | paired delta |
|---|---:|---:|---:|
| 1 | 44.300 | 42.719 | -1.580 |
| 2 | 44.461 | 43.470 | -0.991 |
| 3 | 44.296 | 43.038 | -1.258 |

Median paired delta -1.258 ms/token. SSE content-gap p95: 55.086→53.257 / 56.224→53.525 / 54.670→53.407 ms. p99: 60.962→59.416 / 61.599→59.199 / 61.822→59.711. These are not engine-step percentiles.

Text SHA256 matches across arms and all restarts at each request position. First-vs-reuse text differs, so global all-text-equal is false; request positions must not be pooled for equality.

Separate diagnostic config cpu-first-second-diagnostic.json, /tmp/attr-cpufirst-second-diag-01. Three paired processes each have 18336 exact work records, ALL matching pairwise (IDs, weight bits, residency, final GPU/CPU assignments). Summary cpu-first-second-diagnostic.json. Diagnostic TPOT not used. The record count differs from the database prompt because call shapes/history differ; do not label records decode steps. Activation and physical-byte identity still not directly measured.

## Database prompt: lookahead enabled

Config cpu-first-lookahead-on.json; raw /tmp/attr-cpufirst-lookon-01. Archived cpu-first-lookon-{manifest,tails}.json.

| round | baseline ms/token | CPU-first | paired delta |
|---|---:|---:|---:|
| 1 | 45.063 | 44.949 | -0.114 |
| 2 | 45.626 | 45.025 | -0.601 |
| 3 | 45.797 | 45.177 | -0.620 |

Median paired delta -.601 ms/token. All request positions have text divergence across processes: this is a scheduling-policy end-to-end comparison, NOT strict same-work attribution. Lookahead timing may alter workload. SSE p95 is slightly worse in round 1 (55.499→55.958), improves in rounds 2/3; p99 improves all three (61.566→59.739 / 62.483→60.188 / 62.011→61.235). Do not claim uniformly better tail behavior.

## Decision / remaining gates

Two no-lookahead prompts show consistent scheduling benefit with matched diagnostic records, supporting retention as an opt-in path. Lookahead-on results do not negate the benefit but cannot isolate CPU-start critical-path changes. Do NOT change defaults yet: need event-correlated CPU/GPU boundary measurements, default Rayon/frac .4 tests, and MTP2/chat acceptance. Do not bundle async activation implementation until its independent identity/lifecycle gates pass.
