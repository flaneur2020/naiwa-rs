#!/usr/bin/env python3
"""Multi-round chat driver: replays a fixed transcript against an OpenAI-compatible server and
timestamps every streamed token.

The transcript grows with each engine's *own* streamed answer, exactly as a chat client does, so the
prefix/KV cache is exercised the way a live session exercises it. (A fixed transcript of authored
assistant turns cannot do that: mistralrs snapshots the recurrent prefix at the end of a *prompt*,
so a request whose shared prefix ends in authored text never matches a snapshot, and naiwa measured
zero reuse over seven turns with one. Strata's checkpoint would hide that difference.) Turn 1 is
identical on both engines, which is where prompt-token equality is checked; later turns are compared
by reuse ratio, TPOT and per-turn prompt size rather than by identical text.

Protocol (see README.md): one engine at a time, cold process per round, one discarded warmup
conversation, rounds alternate engine order, concurrency 1.

    run.py --engines naiwa,strata --rounds 3 --corpus corpus.jsonl --out DIR [--classes short,code]
           [--timeout S]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import os
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_ENGINE_DIR = HERE / "engines"

# The discarded warmup conversation: two short turns, never measured. It absorbs model load (a
# server that answers /v1/models before the weights are resident blocks here instead of on turn 1)
# and leaves the prefix cache holding text that no measured conversation shares.
WARMUP = [
    {"user": "Reply with the single word: ready.", "max_tokens": 8},
    {"user": "Now reply with the single word: warm.", "max_tokens": 8},
]


# ---------------------------------------------------------------------------
# corpus / config
# ---------------------------------------------------------------------------


def expand(path: str) -> Path:
    return Path(ospath(path))


def ospath(value: str) -> str:
    """`~` and `$VAR` in a launch argument: the engine is exec'd directly, so no shell expands them."""
    return os.path.expandvars(os.path.expanduser(str(value)))


def load_corpus(path: Path) -> list[dict]:
    """JSONL; an optional first line `{"_constants": {...}}` supplies `{{NAME}}` substitutions."""
    constants: dict[str, str] = {}
    convs: list[dict] = []
    for line_no, raw in enumerate(path.read_text().splitlines(), 1):
        raw = raw.strip()
        if not raw or raw.startswith("#"):
            continue
        rec = json.loads(raw)
        if "_constants" in rec:
            constants.update(rec["_constants"])
            continue
        if "_comment" in rec and "turns" not in rec:
            continue
        text = json.dumps(rec)
        for key, value in constants.items():
            text = text.replace("{{%s}}" % key, json.dumps(value)[1:-1])
        rec = json.loads(text)
        turns = rec["turns"]
        if not turns:
            raise SystemExit(f"{path}:{line_no}: conversation {rec.get('id')!r} has no turns")
        for i, turn in enumerate(turns):
            if "user" not in turn:
                raise SystemExit(f"{path}:{line_no}: turn {i} has no user message")
            turn.setdefault("max_tokens", 96)
        rec.setdefault("system", None)
        rec.setdefault("class", "short")
        convs.append(rec)
    if not convs:
        raise SystemExit(f"{path}: no conversations")
    return convs


def load_engine(path: Path) -> dict:
    cfg = json.loads(path.read_text())
    for key in ("name", "exe", "args", "port"):
        if key not in cfg:
            raise SystemExit(f"{path}: missing {key!r}")
    cfg.setdefault("cwd", ".")
    cfg.setdefault("arm", cfg["name"])
    cfg.setdefault("body", {})
    cfg["_path"] = str(path)
    return cfg


def engine_config(name: str, engine_dir: Path) -> Path:
    candidate = Path(name)
    if candidate.suffix == ".json" and candidate.exists():
        return candidate
    return engine_dir / f"{name}.json"


# ---------------------------------------------------------------------------
# process / server
# ---------------------------------------------------------------------------


def free_port(preferred: int) -> int:
    """`preferred`, or the next free port: two runs must never share one."""
    for port in range(preferred, preferred + 200):
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise SystemExit(f"no free port near {preferred}")


def pgid_rss_mib(pgid: int) -> int:
    """Resident set of the whole process group, in MiB (the engine may be a child of the server)."""
    total = 0
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text()
        except OSError:
            continue
        # comm may contain spaces and parentheses: fields start after the last ')'.
        fields = stat[stat.rfind(")") + 2 :].split()
        try:
            if int(fields[2]) != pgid:  # pgrp is field 5 overall, 3rd after comm/state
                continue
            pages = int((entry / "statm").read_text().split()[1])
        except (OSError, IndexError, ValueError):
            continue
        total += pages * os.sysconf("SC_PAGE_SIZE")
    return total >> 20


class Sampler(threading.Thread):
    """1 Hz VRAM + RSS peaks while a round runs."""

    def __init__(self, pgid_getter):
        super().__init__(daemon=True)
        self._pgid = pgid_getter
        self.stop = threading.Event()
        self.vram = 0
        self.rss = 0
        self.samples = 0

    def run(self) -> None:
        while not self.stop.is_set():
            self.samples += 1
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                ).stdout.split()
                self.vram = max(self.vram, int(out[0]))
            except Exception:
                pass
            pgid = self._pgid()
            if pgid:
                try:
                    self.rss = max(self.rss, pgid_rss_mib(pgid))
                except Exception:
                    pass
            self.stop.wait(1.0)


class Engine:
    def __init__(self, cfg: dict, out_dir: Path, round_no: int):
        self.cfg = cfg
        self.port = free_port(int(cfg["port"]))
        self.base = f"http://127.0.0.1:{self.port}"
        fill = {"port": self.port, "out": out_dir, "round": round_no, "name": cfg["name"]}
        # An engine that takes its launch config as a file (strata) gets a per-round copy, so the
        # port and the log path cannot collide with a previous round.
        config_path = None
        if cfg.get("write_config"):
            spec = cfg["write_config"]
            source = json.loads(expand(spec["from"]).read_text())
            for key, value in (spec.get("set") or {}).items():
                source[key] = str(value).format(**fill) if isinstance(value, str) else value
            config_path = Path(str(spec["path"]).format(**fill))
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(json.dumps(source, indent=2) + "\n")
            fill["config"] = config_path
        args = [ospath(str(x).format(**fill)) for x in cfg["args"]]
        self.args = [ospath(cfg["exe"])] + args
        # Read-only env probes (design.md: env switches are probes, real knobs are flags). Recorded in the
        # launch line so a run can be reproduced from its own log.
        self.env = {str(k): str(v).format(**fill) for k, v in (cfg.get("env") or {}).items()}
        log_dir = out_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = log_dir / f"{cfg['name']}-r{round_no}.log"
        self.proc: subprocess.Popen | None = None
        self.load_s = 0.0

    def start(self) -> None:
        self.log = open(self.log_path, "w")
        env = " ".join(f"{k}={v}" for k, v in sorted(self.env.items()))
        self.log.write("# " + (env + " " if env else "") + " ".join(self.args) + "\n")
        self.log.flush()
        self.proc = subprocess.Popen(
            self.args,
            cwd=str(expand(self.cfg["cwd"])),
            env={**os.environ, **self.env} if self.env else None,
            stdout=self.log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,  # one process group: SIGTERM then SIGKILL reach the engine too
        )

    def wait_ready(self, timeout_s: float) -> str:
        """Poll /v1/models; return the model id both servers want in the request body."""
        start = time.perf_counter()
        deadline = start + timeout_s
        while time.perf_counter() < deadline:
            if self.proc.poll() is not None:
                raise SystemExit(
                    f"{self.cfg['name']}: exited with {self.proc.returncode}; see {self.log_path}"
                )
            try:
                with urllib.request.urlopen(self.base + "/v1/models", timeout=10) as r:
                    data = json.loads(r.read())
                self.load_s = time.perf_counter() - start
                return data["data"][0]["id"]
            except (urllib.error.URLError, OSError, KeyError, json.JSONDecodeError):
                time.sleep(0.5)
        raise SystemExit(f"{self.cfg['name']}: not ready after {timeout_s:.0f}s; see {self.log_path}")

    def stop(self) -> None:
        if self.proc is None:
            return
        try:
            pgid = os.getpgid(self.proc.pid)
        except OSError:
            return
        for sig, grace in ((signal.SIGTERM, 25), (signal.SIGKILL, 5)):
            try:
                os.killpg(pgid, sig)
            except OSError:
                break
            try:
                self.proc.wait(timeout=grace)
                break
            except subprocess.TimeoutExpired:
                continue
        self.log.close()


# ---------------------------------------------------------------------------
# one streamed turn
# ---------------------------------------------------------------------------


# The engine's Prometheus counters, read before and after each turn so a turn gets its own deltas. The server is
# single-flight for the harness's requests, so the difference is attributable. Names match mistralrs-core's
# `engine/logger.rs` (`add_speculative_stats`); a turn without speculation reports zeros, an engine without the
# endpoint (strata) reports None and the columns stay empty rather than invented.
SPEC_COUNTERS = (
    "mistralrs_speculative_drafts_total",  # verification steps: one per decode step
    "mistralrs_speculative_draft_tokens_proposed_total",
    "mistralrs_speculative_draft_tokens_accepted_total",
)
# Per drafted position, how often it survived: the shape of acceptance (position 0 is the first draft).
SPEC_POS = "mistralrs_speculative_draft_tokens_accepted_per_pos_total"
SPEC_POS_RE = re.compile(r'^' + SPEC_POS + r'\{position="(\d+)"\} ')


def read_metrics(base: str) -> dict:
    try:
        with urllib.request.urlopen(base + "/metrics", timeout=5) as r:
            text = r.read().decode("utf-8", "replace")
    except Exception:  # noqa: BLE001 - no endpoint is a fact about the engine, not an error
        return {}
    got = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name, _, value = line.partition(" ")
        if name in SPEC_COUNTERS:
            try:
                got[name] = float(value)
            except ValueError:
                pass
            continue
        m = SPEC_POS_RE.match(line)
        if m:
            try:
                got[f"{SPEC_POS}:{int(m.group(1))}"] = float(value)
            except ValueError:
                pass
    return got


def spec_delta(before: dict, after: dict, completion_tokens) -> dict:
    """Steps, drafts, acceptance and tokens per step for one turn."""
    if not before or not after:
        return {"spec_steps": None, "spec_proposed": None, "spec_accepted": None,
                "acceptance": None, "tokens_per_step": None}
    steps = after.get(SPEC_COUNTERS[0], 0.0) - before.get(SPEC_COUNTERS[0], 0.0)
    proposed = after.get(SPEC_COUNTERS[1], 0.0) - before.get(SPEC_COUNTERS[1], 0.0)
    accepted = after.get(SPEC_COUNTERS[2], 0.0) - before.get(SPEC_COUNTERS[2], 0.0)
    tokens = completion_tokens - 1 if isinstance(completion_tokens, int) else None
    # Where the drafts die: position 0 is the first draft, so this says whether the drafter is weak everywhere or
    # only deep. A step commits `accepted + 1` tokens, which is why acceptance divides TPOT at a fixed step time.
    positions = []
    for i in range(8):
        key = f"{SPEC_POS}:{i}"
        if key not in after:
            break
        delta = after.get(key, 0.0) - before.get(key, 0.0)
        positions.append(delta / steps if steps > 0 else None)
    return {
        "spec_steps": int(steps),
        "spec_proposed": int(proposed),
        "spec_accepted": int(accepted),
        "acceptance": (accepted / proposed) if proposed > 0 else None,
        "tokens_per_step": (tokens / steps) if steps > 0 and tokens else None,
        "accept_pos": positions or None,
    }


def stream_turn(base: str, body: dict, timeout: float) -> dict:
    req = urllib.request.Request(
        base + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    marks: list[float] = []
    answer: list[str] = []
    content_chars = 0
    reasoning_chars = 0
    usage = None
    finish = None
    t_last = None
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            event = json.loads(payload)
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices") or []:
                delta = choice.get("delta") or {}
                reason = delta.get("reasoning_content") or ""
                text = delta.get("content") or ""
                if reason:
                    reasoning_chars += len(reason)
                if text:
                    content_chars += len(text)
                    answer.append(text)
                    marks.append(time.perf_counter())
                if choice.get("finish_reason"):
                    finish = choice["finish_reason"]
    if marks:
        t_last = marks[-1]
        ttft = marks[0] - t0
        itls = [b - a for a, b in zip(marks, marks[1:])]
        decode = (len(marks) - 1) / (t_last - marks[0]) if len(marks) > 1 and t_last > marks[0] else None
        # A speculative decoder delivers several tokens per step, so most deltas arrive in a burst
        # and the ITL median is 0. TPOT is the honest average: decode window / gaps between tokens.
        tpot = (t_last - marks[0]) / (len(marks) - 1) if len(marks) > 1 else None
        e2e = t_last - t0
    else:
        # An answer that produced no content delta: keep the failure visible, do not invent a number.
        ttft = decode = tpot = e2e = None
        itls = []
    cached = None
    prompt_tokens = None
    completion_tokens = None
    if usage:
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    return {
        "answer": "".join(answer),
        "ttft_ms": None if ttft is None else ttft * 1e3,
        "itl_p50_ms": statistics.median(itls) * 1e3 if itls else None,
        "itl_p90_ms": percentile(itls, 90) * 1e3 if itls else None,
        "tpot_ms": None if tpot is None else tpot * 1e3,
        "decode_tps": decode,
        "e2e_ms": None if e2e is None else e2e * 1e3,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens
        if completion_tokens is not None
        else len(marks),
        "cached_tokens": cached,
        "reuse": (cached / prompt_tokens) if cached is not None and prompt_tokens else None,
        "n_deltas": len(marks),
        "content_chars": content_chars,
        "reasoning_chars": reasoning_chars,
        "finish": finish,
    }


def percentile(values: list[float], pct: float) -> float:
    if not values:
        raise ValueError("empty")
    ordered = sorted(values)
    k = (len(ordered) - 1) * pct / 100.0
    lo = int(k)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def conversation_messages(conv: dict, upto: int, answers: list[str]) -> list[dict]:
    """The transcript for turn `upto`: the corpus's user turns plus the answers this engine gave."""
    messages = []
    if conv.get("system"):
        messages.append({"role": "system", "content": conv["system"]})
    for i in range(upto):
        messages.append({"role": "user", "content": conv["turns"][i]["user"]})
        messages.append({"role": "assistant", "content": answers[i]})
    messages.append({"role": "user", "content": conv["turns"][upto]["user"]})
    return messages


# ---------------------------------------------------------------------------
# one round
# ---------------------------------------------------------------------------


def run_round(cfg: dict, convs: list[dict], out_dir: Path, round_no: int, args) -> None:
    # A load-time CUDA OOM on a 12 GB card is intermittent (the model's PLE table falls back to a
    # host gather and the allocator fragments); that is not a measurement, so startup is retried.
    for attempt in range(args.start_retries + 1):
        try:
            _run_round_once(cfg, convs, out_dir, round_no, args)
            return
        except StartupFailed as e:
            if attempt == args.start_retries:
                raise
            print(f"  {cfg['name']}: startup failed ({e}); retry {attempt + 1}", flush=True)


class StartupFailed(RuntimeError):
    pass


def _run_round_once(cfg: dict, convs: list[dict], out_dir: Path, round_no: int, args) -> None:
    engine = Engine(cfg, out_dir, round_no)
    sampler = Sampler(lambda: os.getpgid(engine.proc.pid) if engine.proc else None)
    print(f"[{time.strftime('%H:%M:%S')}] round {round_no}: {cfg['name']} on :{engine.port}", flush=True)
    engine.start()
    sampler.start()
    try:
        try:
            model = engine.wait_ready(args.ready_timeout)
        except SystemExit as e:
            raise StartupFailed(str(e)) from e
        print(f"  ready in {engine.load_s:.1f}s ({model})", flush=True)

        def request(messages: list[dict], max_tokens: int) -> dict:
            body = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0,
                "top_p": 1,
                "seed": 0,
                "stream": True,
            }
            body.update(cfg.get("body") or {})
            before = read_metrics(engine.base)
            out = stream_turn(engine.base, body, args.timeout)
            after = read_metrics(engine.base)
            out.update(spec_delta(before, after, out.get("completion_tokens")))
            return out

        for turn in WARMUP:
            request([{"role": "user", "content": turn["user"]}], turn["max_tokens"])

        with (out_dir / "turns.jsonl").open("a") as sink:
            for conv in convs:
                answers: list[str] = []
                for i, turn in enumerate(conv["turns"]):
                    rec = {
                        "round": round_no,
                        "engine": cfg["name"],
                        "arm": cfg["arm"],
                        "conv": conv["id"],
                        "class": conv["class"],
                        "turn": i + 1,
                        "max_tokens": turn["max_tokens"],
                    }
                    try:
                        rec.update(
                            request(
                                conversation_messages(conv, i, answers), turn["max_tokens"]
                            )
                        )
                        if rec["n_deltas"] == 0:
                            # The server closed the stream without a token: an OOM or an engine
                            # error, not a short answer. Data, but the round cannot continue.
                            raise RuntimeError(f"no content delta (finish {rec['finish']!r})")
                    except Exception as e:  # noqa: BLE001 - a failed turn is data, not a crash
                        rec["error"] = f"{type(e).__name__}: {e}"
                        sink.write(json.dumps(rec) + "\n")
                        sink.flush()
                        raise SystemExit(
                            f"{cfg['name']} round {round_no} {conv['id']} turn {i + 1} failed: {e}"
                        ) from e
                    answers.append(rec.pop("answer", ""))
                    sink.write(json.dumps(rec) + "\n")
                    sink.flush()
                    print(
                        f"  {conv['class']:5s} {conv['id']:8s} t{i + 1} "
                        f"ttft {fmt(rec['ttft_ms'])} tpot {fmt(rec['tpot_ms'])} "
                        f"tok/s {fmt(rec.get('decode_tps'))} reuse {fmt_ratio(rec['reuse'])} "
                        f"accept {fmt_ratio(rec.get('acceptance'))} tok/step {fmt(rec.get('tokens_per_step'))}",
                        flush=True,
                    )
    finally:
        sampler.stop.set()
        sampler.join(timeout=2)
        engine.stop()
        with (out_dir / "rounds.jsonl").open("a") as sink:
            sink.write(
                json.dumps(
                    {
                        "round": round_no,
                        "engine": cfg["name"],
                        "arm": cfg["arm"],
                        "load_s": engine.load_s,
                        "peak_vram_mib": sampler.vram,
                        "peak_rss_mib": sampler.rss,
                        "samples": sampler.samples,
                        "log": str(engine.log_path),
                    }
                )
                + "\n"
            )


def fmt(x) -> str:
    return "-" if x is None else f"{x:.0f}"


def fmt_ratio(x) -> str:
    return "-" if x is None else f"{x:.2f}"


# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engines", default="naiwa,strata", help="engine config names, run in this order per round")
    ap.add_argument("--engine-dir", default=str(DEFAULT_ENGINE_DIR))
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--corpus", default=str(HERE / "corpus.jsonl"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default=None, help="only conversations of this class (short/doc/code/long)")
    ap.add_argument("--classes", default=None, help="comma-separated class filter, e.g. short,code")
    ap.add_argument("--timeout", type=float, default=900.0, help="per-request HTTP timeout")
    ap.add_argument("--ready-timeout", type=float, default=900.0, help="server start → /v1/models")
    ap.add_argument("--start-retries", type=int, default=2, help="retries when the engine fails to start")
    args = ap.parse_args()

    corpus_path = expand(args.corpus)
    convs = load_corpus(corpus_path)
    wanted = set()
    if args.only:
        wanted.add(args.only)
    if args.classes:
        wanted.update(x.strip() for x in args.classes.split(",") if x.strip())
    if wanted:
        convs = [c for c in convs if c["class"] in wanted]
        if not convs:
            raise SystemExit(f"no conversation of class {sorted(wanted)}")
    configs = [load_engine(engine_config(n, expand(args.engine_dir))) for n in args.engines.split(",")]

    out_dir = expand(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run.json").write_text(
        json.dumps(
            {
                "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "corpus": str(corpus_path),
                "corpus_sha256": hashlib.sha256(corpus_path.read_bytes()).hexdigest(),
                "classes": sorted({c["class"] for c in convs}),
                "conversations": [c["id"] for c in convs],
                "rounds": args.rounds,
                "concurrency": 1,
                "warmup_turns": len(WARMUP),
                "engines": [
                    {
                        k: v
                        for k, v in cfg.items()
                        if k not in ("_path",)
                    }
                    for cfg in configs
                ],
            },
            indent=2,
        )
        + "\n"
    )

    for round_no in range(1, args.rounds + 1):
        order = configs if round_no % 2 == 1 else list(reversed(configs))
        for cfg in order:
            run_round(cfg, convs, out_dir, round_no, args)
    print(f"[{time.strftime('%H:%M:%S')}] done: {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
