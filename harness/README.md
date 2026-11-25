# harness

Driver scripts and measurement instruments that are not part of the engine. Everything here is
runnable from the repository root and writes its logs under `$OUT` (default `~/wtmp`).

| path | what it is |
|---|---|
| `m1/` | frozen greedy oracle gate (`capture_oracle.sh`, `reference.oracle.txt`) |
| `attribution/` | restarted clean/diagnostic A/B, exact MoE-work comparison, CPU window accounting, cross-stream GPU unions, and unit tests |
| `serve/` | launch profiles: `serve-tuned-mtp2.sh` (fastest verified MTP2), `serve.sh`, `serve-pi.sh` (16k interactive coding agent) |
| `chat/` | interactive REPL (`chat.sh`, `cli.py`) and multi-turn chat benchmark (`run.py`, `summarize.py`, `engines/*.json`) |
| `tpot/` | time per output token of a running server (`tpot.py`, `ladder.py`), median of N greedy rounds |
| `bench/run_with_mem.sh` | 1 Hz RAM/VRAM sampler around any run |
| `profile/nsys_engine.sh` | profile an engine under `nsys` |

Two habits these encode, because both cost time to learn: the number that gets quoted is a
**median of at least three rounds**, never a mean; and a memory or liveness claim comes from a
sampler (`run_with_mem.sh`), not from the allocator reporting on itself.
