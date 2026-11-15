#!/usr/bin/env python3
r"""Interactive chat against a running naiwa-rs `/v1/chat/completions`.

`chat.sh` starts the engine and then exec's this; run it directly (with `--base-url`) when a server
is already up. Stdlib only, on purpose: the harness has to run on this box without a venv, and all a
chat client needs from the engine is the SSE stream `run.py` already parses.

    harness/chat/chat.sh                          # engine + REPL, one line
    harness/chat/cli.py --base-url http://127.0.0.1:8095/v1
    harness/chat/cli.py --once "what is a MoE router?"      # one turn, then exit

Slash commands in the REPL: /help /reset /system /temp /max /history /save /exit.
A line ending in `\` continues on the next one; Ctrl-C cancels the *turn* (the conversation
survives), Ctrl-D or /exit ends it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

try:  # line editing + arrow-key history; absent only on a stripped-down python build
    import readline
except ImportError:  # pragma: no cover
    readline = None

HISTORY_FILE = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "naiwa-chat" / "history"

COLOR = sys.stdout.isatty() and not os.environ.get("NO_COLOR")


def dim(text: str) -> str:
    return f"\033[2m{text}\033[0m" if COLOR else text


def red(text: str) -> str:
    return f"\033[31m{text}\033[0m" if COLOR else text


class ChatError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# session
# ---------------------------------------------------------------------------


class Session:
    """Conversation state plus the knobs a slash command can change mid-session."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.base_url = args.base_url.rstrip("/")
        self.model = args.model
        self.max_tokens = args.max_tokens
        self.temperature = args.temperature
        self.thinking = args.thinking
        self.timeout = args.timeout
        self.messages: list[dict] = []
        self.system = args.system or ""
        if self.system:
            self.messages.append({"role": "system", "content": self.system})

    def set_system(self, text: str) -> None:
        self.system = text
        others = [m for m in self.messages if m["role"] != "system"]
        self.messages = ([{"role": "system", "content": text}] if text else []) + others

    def reset(self) -> None:
        self.messages = []
        if self.system:
            self.messages.append({"role": "system", "content": self.system})

    def turn(self, text: str) -> tuple[str, dict]:
        """One user turn, streamed to stdout. Returns (answer, stats)."""
        self.messages.append({"role": "user", "content": text})
        body = {
            "model": self.model,
            "messages": self.messages,
            "stream": True,
            # Without this the engine only reports token counts for non-streamed requests, and the
            # cost line after each turn is the only feedback on cache reuse a human can act on.
            "stream_options": {"include_usage": True},
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            # mistral-rs extension, not OpenAI: off keeps the reasoning block from eating the whole
            # token budget on a Qwen3-family template. `--thinking` turns it back on.
            "enable_thinking": self.thinking,
        }
        req = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        )
        t0 = time.perf_counter()
        ttft = None
        content: list[str] = []
        reasoning: list[str] = []
        other: list[str] = []
        usage: dict | None = None
        finish: str | None = None

        try:
            resp = urllib.request.urlopen(req, timeout=self.timeout)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:800]
            raise ChatError(f"HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, OSError) as exc:
            raise ChatError(f"cannot reach {self.base_url}: {exc}") from exc

        with resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line or line.startswith((":", "event:", "id:", "retry:")):
                    continue
                if not line.startswith("data:"):
                    # A non-SSE line is the engine's own message, not a token: keep it for the error.
                    other.append(line)
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                event = json.loads(payload)
                # A failed step arrives as an `error` payload on the same stream. Dropping it would
                # surface as an empty answer, which is the one failure mode a chat client must not have.
                if event.get("error"):
                    raise ChatError(f"server error: {json.dumps(event['error'])[:800]}")
                if event.get("usage"):
                    usage = event["usage"]
                if ttft is None and event.get("choices"):
                    ttft = time.perf_counter() - t0
                for choice in event.get("choices") or []:
                    if choice.get("finish_reason"):
                        finish = choice["finish_reason"]
                    delta = choice.get("delta") or {}
                    if delta.get("reasoning_content"):
                        reasoning.append(delta["reasoning_content"])
                        sys.stdout.write(dim(delta["reasoning_content"]))
                        sys.stdout.flush()
                    if delta.get("content"):
                        if reasoning and not content:
                            sys.stdout.write("\n\n")
                        content.append(delta["content"])
                        sys.stdout.write(delta["content"])
                        sys.stdout.flush()

        answer = "".join(content)
        elapsed = time.perf_counter() - t0
        if answer or finish:
            # Reasoning is display-only: this model runs with thinking pinned off, and replaying a
            # template-specific block would make the next prompt depend on the chat template.
            self.messages.append({"role": "assistant", "content": answer})
        # `stream_options.include_usage` is what makes this the engine's own count; `len(content)`
        # is only a fallback for a server that never sends a usage event (it counts deltas, not tokens).
        ctoks = (usage or {}).get("completion_tokens") or len(content)
        stats = {
            "completion_tokens": ctoks,
            "elapsed": elapsed,
            "tok_s": ctoks / elapsed if elapsed > 0 else 0.0,
            "ttft_ms": (ttft or 0.0) * 1000.0,
            "prompt_tokens": (usage or {}).get("prompt_tokens"),
            "cached_tokens": ((usage or {}).get("prompt_tokens_details") or {}).get("cached_tokens"),
            "finish": finish,
            "other": other,
        }
        return answer, stats


def cost_line(stats: dict) -> str:
    parts = [
        f"{stats['completion_tokens']} tok",
        f"{stats['elapsed']:.1f}s",
        f"{stats['tok_s']:.1f} tok/s",
        f"ttft {stats['ttft_ms']:.0f} ms",
    ]
    if stats["prompt_tokens"] is not None:
        cached = stats["cached_tokens"] or 0
        parts.append(f"prompt {stats['prompt_tokens']} (cached {cached})")
    if stats["finish"]:
        parts.append(stats["finish"])
    return dim("[ " + " · ".join(parts) + " ]")


# ---------------------------------------------------------------------------
# REPL
# ---------------------------------------------------------------------------

HELP = """\
/help            this text
/reset           forget the conversation (keeps the system prompt)
/system [text]   set (or show, with no argument) the system prompt
/temp <float>    sampling temperature (0 = greedy)
/max <int>       max_tokens for the next turns
/history         print the conversation so far
/save <path>     write the conversation as JSON
/exit            leave (Ctrl-D does the same)
Anything else is sent as a user turn; end a line with `\\` to continue it."""


def read_user_input() -> str | None:
    """One user message. Returns None on EOF. A trailing backslash continues the line."""
    try:
        line = input(">>> ")
        while line.endswith("\\"):
            line = line[:-1] + "\n" + input("... ")
        return line
    except EOFError:
        return None


def handle_command(session: Session, line: str) -> bool:
    """Returns False when the REPL should end."""
    cmd, _, rest = line.partition(" ")
    rest = rest.strip()
    if cmd in ("/exit", "/quit", "/q"):
        return False
    if cmd == "/help":
        print(HELP)
    elif cmd == "/reset":
        session.reset()
        print(dim("[ conversation cleared ]"))
    elif cmd == "/system":
        if rest:
            session.set_system(rest)
        print(dim(f"[ system prompt: {session.system or '(none)'} ]"))
    elif cmd == "/temp":
        try:
            if rest:
                session.temperature = float(rest)
        except ValueError:
            print(red(f"not a number: {rest}"))
            return True
        print(dim(f"[ temperature {session.temperature} ]"))
    elif cmd == "/max":
        try:
            if rest:
                session.max_tokens = int(rest)
        except ValueError:
            print(red(f"not an integer: {rest}"))
            return True
        print(dim(f"[ max_tokens {session.max_tokens} ]"))
    elif cmd == "/history":
        for msg in session.messages:
            print(f"{msg['role']:>9}: {msg['content'][:400]}")
    elif cmd == "/save":
        path = Path(rest or "chat.json").expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(session.messages, indent=2) + "\n")
        print(dim(f"[ saved {path} ]"))
    else:
        print(red(f"unknown command {cmd}; try /help"))
    return True


def repl(session: Session) -> int:
    if readline is not None:
        try:
            HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
            readline.read_history_file(HISTORY_FILE)
        except OSError:
            pass
        readline.set_history_length(1000)

    print(dim(f"[ {session.base_url} · model {session.model} · /help for commands ]"))
    while True:
        try:
            line = read_user_input()
        except KeyboardInterrupt:
            print()
            continue
        if line is None:
            print()
            break
        if not line.strip():
            continue
        if line.startswith("/"):
            if not handle_command(session, line):
                break
            continue
        try:
            _, stats = session.turn(line)
        except KeyboardInterrupt:
            # The turn is dropped rather than kept half-answered: a truncated assistant message
            # would be replayed to the engine on every later turn.
            session.messages.pop()
            print(dim("\n[ cancelled ]"))
            continue
        except ChatError as exc:
            session.messages.pop()
            print(red(f"\n[ {exc} ]"))
            continue
        print("\n" + cost_line(stats))
        for line in stats["other"]:
            # Something arrived outside the SSE framing (a proxy page, a warning). It is the first
            # clue when an answer is empty, so it is shown instead of dropped.
            print(dim(f"[ server: {line[:300]} ]"))

    if readline is not None:
        try:
            readline.write_history_file(HISTORY_FILE)
        except OSError:
            pass
    return 0


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8095/v1"))
    ap.add_argument("--model", default=os.environ.get("NAIWA_MODEL", "default"),
                    help="`default` targets the only loaded model")
    ap.add_argument("--system", default=os.environ.get("CHAT_SYSTEM", ""))
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--thinking", action="store_true", help="let the model emit its reasoning block")
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--once", metavar="PROMPT",
                    help="one turn and exit (prompt from stdin when PROMPT is `-`)")
    args = ap.parse_args(argv)

    session = Session(args)
    if args.once is None:
        return repl(session)

    prompt = sys.stdin.read() if args.once == "-" else args.once
    try:
        answer, stats = session.turn(prompt)
    except ChatError as exc:
        print(red(f"[ {exc} ]"), file=sys.stderr)
        return 1
    if answer and not answer.endswith("\n"):
        print()
    # stdout is a pipe here (not a tty) and stderr is not: flush so the two streams do not interleave.
    sys.stdout.flush()
    for line in stats["other"]:
        print(dim(f"[ server: {line[:300]} ]"), file=sys.stderr)
    print(cost_line(stats), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
