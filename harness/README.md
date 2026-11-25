# harness

Driver scripts and measurement instruments that are not part of the engine. Everything here is
runnable from the repository root and writes its logs under `$OUT` (default `~/wtmp`).

| path | what it is |
|---|---|
| `oracle/` | frozen greedy oracle gate (`capture_oracle.sh`, `reference.oracle.txt`) |
| `serve/` | launch profiles: `serve-tuned-mtp2.sh` (fastest verified MTP2), `serve.sh`, `serve-pi.sh` (16k interactive coding agent) |
| `attribution/` | restarted clean/diagnostic A/B (`run.py`, `analyze.py`, `configs/*.json`), exact MoE-work diff, GPU/CPU attribution, and unit tests |
| `chat/` | interactive REPL (`chat.sh`, `cli.py`) and multi-turn chat benchmark (`run.py`, `summarize.py`, `engines/*.json`) |
| `tpot/` | single-server TPOT & ladder benchmark (`tpot.py`, `ladder.py`), median of N greedy rounds |
| `profile/` | 1 Hz RAM/VRAM sampler (`run_with_mem.sh`) and `nsys` profiler (`nsys_engine.sh`) |

Two habits these encode, because both cost time to learn: the number that gets quoted is a
**median of at least three rounds**, never a mean; and a memory or liveness claim comes from a
sampler (`run_with_mem.sh`), not from the allocator reporting on itself.
