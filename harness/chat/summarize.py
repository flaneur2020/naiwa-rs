#!/usr/bin/env python3
"""Summarise a `run.py` directory into the markdown tables that go into the milestone.

    summarize.py --out ~/bench-rel/m10/2026-10-19 [--markdown]

Groups are engine x class. Turn 1 is reported separately from turns 2..N because it has no prefix
to reuse; that split *is* the chat result. Everything is a median over turns; `--rounds` rounds are
pooled, and the per-round extremes are printed so a warm process or a slow round is visible.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def med(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def pct(values, p):
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    k = (len(values) - 1) * p / 100.0
    lo, hi = int(k), min(int(k) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def f(value, digits=0) -> str:
    if value is None:
        return "–"
    return f"{value:.{digits}f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--markdown", action="store_true", help="print the document section only")
    args = ap.parse_args()
    out = Path(args.out).expanduser()
    run = json.loads((out / "run.json").read_text())
    turns = load_jsonl(out / "turns.jsonl")
    rounds = load_jsonl(out / "rounds.jsonl")
    if not turns:
        raise SystemExit(f"{out}: no turns")

    # A run directory may hold several invocations (e.g. short+code for 3 rounds, then doc+long
    # for 1); `run.json` describes the last one, so the index comes from the turns themselves.
    engines = sorted({t["engine"] for t in turns}, key=lambda e: [t for t in turns if t["engine"] == e][0]["round"])
    classes = sorted({t["class"] for t in turns})
    rounds_run = sorted({t["round"] for t in turns})
    failed = [t for t in turns if t.get("error")]
    first = [t for t in turns if t["turn"] == 1]
    later = [t for t in turns if t["turn"] > 1]

    def group(rows, engine, cls):
        return [t for t in rows if t["engine"] == engine and t["class"] == cls]

    print(f"# M10.2 — multi-round chat, naiwa-rs vs Strata\n")
    print(f"- run: `{out}` · corpus `{run['corpus_sha256'][:12]}` · classes {classes} · "
          f"rounds {rounds_run} · concurrency {run['concurrency']}")
    run_engines = {e["name"]: e for e in run["engines"]}
    print(f"- engines: " + ", ".join(
        f"**{name}** ({run_engines.get(name, {}).get('arm', '?')})" for name in engines))
    if failed:
        print(f"- **{len(failed)} failed turn(s)**: " + ", ".join(
            f"{t['engine']}/{t['conv']}/t{t['turn']} {t['error']}" for t in failed[:5]))
    print()

    # 1. per class -------------------------------------------------------------------------------
    print("## Speculation (acceptance divides TPOT at a fixed step time: a step commits accepted + 1 tokens)")
    print()
    print("| class | engine | accept p50 | tok/step p50 | accepted per drafted position |")
    print("|---|---|---:|---:|---|")
    for cls in classes:
        for engine in engines:
            late = group(later, engine, cls)
            if not late or all(t.get("acceptance") is None for t in late):
                continue
            # One number per drafted position: the median over turns, not every turn's value (the first version
            # concatenated them and printed 24 columns for 6 turns).
            # An arm recorded before the field existed (or an engine that cannot report it) has no per-position list
            # at all, and a short list (a step that never reached position 3) has fewer entries than 4.
            def pos(i):
                vals = []
                for t in late:
                    row = t.get("accept_pos") or []
                    vals.append(row[i] if i < len(row) else None)
                return med(vals)

            per_pos = " / ".join(f(pos(i), 2) for i in range(4))
            print(
                f"| {cls} | {engine} | {f(med([t.get('acceptance') for t in late]), 2)} | "
                f"{f(med([t.get('tokens_per_step') for t in late]), 2)} | {per_pos} |"
            )
    print()
    print("## Per class\n")
    print("| class | engine | turns | TTFT t1 | TTFT ≥2 p50 | p90 | TPOT p50 | decode tok/s | reuse p50 "
          "| accept | tok/step | e2e/conv |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for cls in classes:
        for engine in engines:
            late = group(later, engine, cls)
            t1 = group(first, engine, cls)
            if not t1 and not late:
                continue
            e2e = defaultdict(float)
            for t in t1 + late:
                e2e[(t["round"], t["conv"])] += t.get("e2e_ms") or 0.0
            print(
                f"| {cls} | {engine} | {len(t1) + len(late)} | "
                f"{f(med([t['ttft_ms'] for t in t1]))} ms | "
                f"{f(med([t['ttft_ms'] for t in late]))} ms | "
                f"{f(pct([t['ttft_ms'] for t in late], 90))} ms | "
                f"{f(med([t['tpot_ms'] for t in late]))} ms | "
                f"{f(med([t['decode_tps'] for t in late]), 1)} | "
                f"{f(med([t['reuse'] for t in late]), 2)} | "
                f"{f(med([t['acceptance'] for t in late]), 2)} | "
                f"{f(med([t['tokens_per_step'] for t in late]), 2)} | "
                f"{f(med(list(e2e.values())) / 1000.0, 1)} s |"
            )
    print("\nTTFT = request write → first content delta. TPOT = decode window / gaps (speculative")
    print("decoders deliver several tokens per step, so an ITL median would read 0). Reuse =")
    print("`prompt_tokens_details.cached_tokens / prompt_tokens`, median over turns ≥2.\n")

    # 2. per turn --------------------------------------------------------------------------------
    for cls in classes:
        print(f"### `{cls}` per turn\n")
        header = "| turn |" + "".join(f" {e} TTFT | {e} TPOT | {e} reuse |" for e in engines)
        print(header)
        print("|---:|" + "---:|" * (3 * len(engines)))
        n_turns = max((t["turn"] for t in turns if t["class"] == cls), default=0)
        for n in range(1, n_turns + 1):
            row = f"| {n} |"
            for engine in engines:
                rows = [t for t in turns if t["engine"] == engine and t["class"] == cls and t["turn"] == n]
                row += (f" {f(med([t['ttft_ms'] for t in rows]))} ms |"
                        f" {f(med([t['tpot_ms'] for t in rows]))} ms |"
                        f" {f(med([t['reuse'] for t in rows]), 2)} |")
            print(row)
        print()

    # 3. prompt growth + pairing -----------------------------------------------------------------
    print("### Prompt growth (median prompt_tokens per turn)\n")
    print("| class | turn | " + " | ".join(engines) + " |")
    print("|---|---:|" + "---:|" * len(engines))
    for cls in classes:
        n_turns = max((t["turn"] for t in turns if t["class"] == cls), default=0)
        for n in range(1, n_turns + 1):
            cells = []
            for engine in engines:
                rows = [t for t in turns if t["engine"] == engine and t["class"] == cls and t["turn"] == n]
                cells.append(f(med([t["prompt_tokens"] for t in rows])))
            print(f"| {cls} | {n} | " + " | ".join(cells) + " |")
    print("\nTurn 1 is the same prompt on both engines; later turns append each engine's own answer,")
    print("so their prompt sizes differ. Compare TPOT and reuse across engines, TTFT within an engine.\n")

    # 4. per-round health ------------------------------------------------------------------------
    print("### Rounds\n")
    print("| round | engine | load | peak VRAM | peak RSS |")
    print("|---:|---|---:|---:|---:|")
    for r in sorted(rounds, key=lambda r: (r["round"], r["engine"])):
        print(f"| {r['round']} | {r['engine']} | {f(r.get('load_s'), 1)} s | "
              f"{f(r.get('peak_vram_mib') / 1024.0, 2)} GiB | {f(r.get('peak_rss_mib') / 1024.0, 2)} GiB |")
    print()

    # 5. comparability ---------------------------------------------------------------------------
    print("### Comparability record\n")
    for name in engines:
        e = run_engines.get(name)
        if e is None:
            print(f"**{name}**: configured in an earlier invocation of this directory")
            print()
            continue
        print(f"**{name}** (arm `{e.get('arm')}`, port {e['port']})")
        for key, value in (e.get("record") or {}).items():
            print(f"- {key}: {value}")
        print()
    print(f"- request body extras: " + ", ".join(
        f"{name} {run_engines.get(name, {}).get('body') or '{}'}" for name in engines))
    print(f"- warmup: {run['warmup_turns']} discarded turns per round, before the corpus")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
