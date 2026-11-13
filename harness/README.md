# harness

Driver scripts and measurement instruments that are not part of the engine. Everything here is
runnable from the repository root and writes its logs under `$OUT` (default `~/wtmp`).

| path | what it is |
|---|---|
| `m0/`, `m1/`, `m6/`, `m7/` | the per-milestone gates: doorbell visibility, the frozen oracle, region capture, pack layout |
| `attribution/` | restarted clean/diagnostic A/B, exact MoE-work comparison, CPU window accounting and cross-stream GPU interval unions |
| `chat/` | multi-class chat comparison against strata; `engines/*.json` are the configs, `run.py` the driver |
| `serve/serve.sh` | the shipped chat configuration as a plain server, for when you want to drive it yourself |
| `serve/serve-pi.sh` | the same engine configured for an interactive agent (16k context, 2 slots, MTP off): the arm `pi --model naiwa/default` talks to |
| `tpot/tpot.py` | time per output token of a running server, median of N greedy rounds |
| `bench/` | kernel microbenchmarks, and `run_with_mem.sh`, a 1 Hz RAM/VRAM sampler around a run |
| `profile/nsys_engine.sh` | profile either engine under nsys (`ncu` is refused on this card) |

Two habits these encode, because both cost time to learn: the number that gets quoted is a
**median of at least three rounds**, never a mean; and a memory or liveness claim comes from a
sampler (`run_with_mem.sh`), not from the allocator reporting on itself.
