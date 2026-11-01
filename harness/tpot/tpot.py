#!/usr/bin/env python3
"""Time-per-output-token of a running naiwa-rs/strata server, over N greedy rounds.

    python3 harness/tpot/tpot.py PORT ROUNDS PROMPT MODEL_ID

Prints the per-round TPOT and the median. The measurement protocol in docs/roadmap.md
asks for a median of at least three rounds and never a mean, so the median is what the
last number is; a round that produced fewer than three tokens is reported as None rather
than being guessed at.

This is deliberately not the chat harness: it is one prompt, one connection, one number,
for comparing two builds (or two flags) on the same machine in under a minute.
"""
import json, sys, time, urllib.request

port = int(sys.argv[1]); n_runs = int(sys.argv[2]); prompt = sys.argv[3] if len(sys.argv) > 3 else "The capital of France is"
url = f"http://127.0.0.1:{port}/v1/completions"
body = json.dumps({"model": sys.argv[4], "prompt": prompt, "max_tokens": 96, "temperature": 0.0,
                   "top_p": 1.0, "seed": 0, "stream": True}).encode()
tpots = []
for _ in range(n_runs):
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    marks = []
    with urllib.request.urlopen(req) as r:
        for line in r:
            if not line.startswith(b"data: "):
                continue
            payload = line[6:].strip()
            if payload == b"[DONE]":
                break
            d = json.loads(payload)
            if d.get("choices") and d["choices"][0].get("text"):
                marks.append(time.perf_counter())
    if len(marks) < 3:
        tpots.append(None); continue
    tpots.append((marks[-1] - marks[0]) / (len(marks) - 1) * 1e3)
good = sorted(t for t in tpots if t)
med = good[len(good)//2] if good else None
print(f"tpot runs {[None if t is None else round(t,2) for t in tpots]} median {None if med is None else round(med,2)} ms")
