#!/usr/bin/env python3
"""A cache-size ladder that can actually resolve a cache step.

`tpot.py` measures one load: five single requests, and the spread inside a load (34-55 ms on
this card) is larger than the effect of 1 GB of expert cache. `ladder.py` fixes the two things
that made the question unanswerable:

1. **Many rounds on one load, not one request per arm.** Each rung is a cold process, then N
   rounds of the same prompt back to back. The rung's number is the median of those rounds and
   its error bar is the interquartile spread *within* that load - i.e. the noise of the thing we
   are measuring, not the noise of the harness.
2. **The mechanism next to the wall clock.** Each rung also reports the expert cache's own
   counters (hits, misses, staged, PCIe bytes), which `serve` prints on SIGTERM when
   `NAIWA_STATS=1` (``src/commands/serve.rs``). Misses a token is a deterministic function of
   the cache size, so it is the low-variance half of the answer: if a bigger cache does not
   move it, the cache is not the constraint and no amount of timing will say otherwise.

    harness/tpot/ladder.py --caches 2.2,2.9,3.2 --rounds 8 [--prompt FILE] [--extra FLAGS]

Reported per rung: TPOT median [p25, p75], decode tok/s, and the cache counters. The comparison
at the end is a difference of medians with the two rungs' spreads printed beside it, so a delta
smaller than the spreads is not read as a result.
"""

import argparse
import json
import os
import re
import signal
import statistics
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))


def start(cache_gb: str, port: int, extra: list[str], log_path: str, env_extra: dict):
    log = open(log_path, "w")
    env = {**os.environ, "NAIWA_STATS": "1", **env_extra}
    args = [os.path.join(ROOT, "harness/serve/serve.sh"), str(port), cache_gb, *extra]
    proc = subprocess.Popen(args, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, start_new_session=True)
    t0 = time.time()
    while time.time() - t0 < 400:
        if proc.poll() is not None:
            raise SystemExit(f"engine exited ({proc.returncode}); see {log_path}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=5) as r:
                return proc, log, json.loads(r.read())["data"][0]["id"], time.time() - t0
        except Exception:
            time.sleep(0.5)
    raise SystemExit(f"engine not ready after 400 s; see {log_path}")


def stop(proc, log):
    """SIGTERM, so `serve` prints its counters; SIGKILL only if it will not go."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=60)
    except Exception:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass
    log.close()


def one_round(port: int, model_id: str, prompt: str, max_tokens: int) -> dict:
    body = json.dumps({
        "model": model_id,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions", data=body,
        headers={"Content-Type": "application/json"})
    marks: list[float] = []
    t0 = time.perf_counter()
    ttft = None
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            ev = json.loads(payload)
            if ev.get("error"):
                raise SystemExit(f"server error: {ev['error']}")
            for ch in ev.get("choices") or []:
                if (ch.get("text") or ""):
                    now = time.perf_counter()
                    if ttft is None:
                        ttft = now - t0
                    marks.append(now)
    if len(marks) < 2:
        return {"tpot": None, "ttft": ttft, "tokens": len(marks), "wall": time.perf_counter() - t0}
    tpot = (marks[-1] - marks[0]) / (len(marks) - 1)
    return {"tpot": tpot * 1e3, "ttft": ttft, "tokens": len(marks),
            "wall": time.perf_counter() - t0}


COUNTERS = re.compile(
    r"expert cache \(process totals\): (?P<rate>[\d.]+)% hit rate \((?P<hits>\d+) hits, "
    r"(?P<misses>\d+) misses, (?P<staged>\d+) staged, (?P<promotions>\d+) promotions\);"
    r".*?--pcie-frac (?P<pcie>\d+) misses, (?P<mib>[\d.]+) MiB")


def rung(cache_gb: str, port: int, rounds: int, prompt: str, max_tokens: int,
         extra: list[str], env_extra: dict) -> dict:
    log_path = f"/tmp/ladder-{cache_gb}.log"
    proc, log, model_id, load_s = start(cache_gb, port, extra, log_path, env_extra)
    try:
        # One discarded warmup: the first request pays the profile seed and the first chunk's one-time blocks.
        one_round(port, model_id, prompt, 8)
        runs = [one_round(port, model_id, prompt, max_tokens) for _ in range(rounds)]
    finally:
        stop(proc, log)
    tpots = [r["tpot"] for r in runs if r["tpot"]]
    text = open(log_path, encoding="utf-8", errors="replace").read()
    m = COUNTERS.search(text)
    counters = m.groupdict() if m else None
    return {
        "cache": cache_gb,
        "load_s": load_s,
        "rounds": len(runs),
        "tpot_med": statistics.median(tpots) if tpots else None,
        "tpot_p25": statistics.quantiles(tpots, n=4)[0] if len(tpots) > 3 else None,
        "tpot_p75": statistics.quantiles(tpots, n=4)[2] if len(tpots) > 3 else None,
        "tpot_min": min(tpots) if tpots else None,
        "tpot_max": max(tpots) if tpots else None,
        "tok_s": 1000.0 / statistics.median(tpots) if tpots else None,
        "counters": counters,
        "raw": [round(t, 2) for t in tpots],
        "wall_s": sum(r["wall"] for r in runs),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--caches", default="2.2,2.9,3.2", help="comma-separated --expert-cache-gb values")
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=96)
    ap.add_argument("--prompt", default=None, help="prompt file (default: the chat harness's long probe)")
    ap.add_argument("--port", type=int, default=19700)
    ap.add_argument("--extra", default="", help="extra serve flags")
    ap.add_argument("--env", action="append", default=[], metavar="K=V",
                    help="environment for the engine, repeatable (e.g. --env NAIWA_PREFILL_BORROW=1)")
    args = ap.parse_args()

    prompt_path = args.prompt or os.path.join(ROOT, "harness/tpot/prompt.txt")
    if not os.path.exists(prompt_path):
        raise SystemExit(f"no prompt at {prompt_path}; pass --prompt")
    prompt = open(prompt_path, encoding="utf-8").read()
    extra = args.extra.split() if args.extra else []
    env_extra = dict(e.split("=", 1) for e in args.env)

    rows = []
    for i, cache in enumerate([c.strip() for c in args.caches.split(",") if c.strip()]):
        r = rung(cache, args.port + i, args.rounds, prompt, args.max_tokens, extra, env_extra)
        rows.append(r)
        med = f"{r['tpot_med']:.2f} ms" if r["tpot_med"] else "no data"
        print(f"[{cache} GB] load {r['load_s']:.0f}s  TPOT {med} "
              f"[{r['tpot_p25']}, {r['tpot_p75']}]  rounds {r['raw']}", flush=True)

    # The counters are **process totals**, so they carry the prompts' own transfers (a prompt step streams every
    # layer, ~23 GiB) as well as decode's. Reporting them a token of output would be nonsense; per measured round,
    # and as a rate over the measured wall time, is what they mean.
    print("\n| cache | TPOT p50 [p25, p75] | decode tok/s | hit rate | PCIe MiB/round | PCIe GB/s |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        c = r["counters"] or {}
        pcie = float(c["mib"]) / max(1, r["rounds"]) if c else None
        gbs = float(c["mib"]) / 1024.0 / r["wall_s"] if c and r["wall_s"] else None
        med = (f"{r['tpot_med']:.2f} ms [{r['tpot_p25']:.2f}, {r['tpot_p75']:.2f}]"
               if r["tpot_med"] else "no data")
        print(f"| {r['cache']} GB | {med} | {r['tok_s'] or 0:.1f} | {c.get('rate', '?')}% "
              f"| {pcie or 0:.0f} | {gbs or 0:.2f} |")

    if len(rows) > 1:
        print("\nDeltas against the first rung (a delta smaller than the spreads is not a result):")
        base = rows[0]
        for r in rows[1:]:
            d = r["tpot_med"] - base["tpot_med"]
            spread = max((r["tpot_p75"] or r["tpot_med"]) - (r["tpot_p25"] or r["tpot_med"]),
                         (base["tpot_p75"] or base["tpot_med"]) - (base["tpot_p25"] or base["tpot_med"]))
            verdict = "resolved" if abs(d) > spread else "inside the spread"
            print(f"  {r['cache']} GB vs {base['cache']} GB: {d:+.2f} ms "
                  f"(spreads {spread:.2f} ms) -> {verdict}")


if __name__ == "__main__":
    sys.exit(main())
