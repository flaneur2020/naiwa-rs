#!/usr/bin/env bash
# Chat with naiwa-rs in one line: start the engine, then hand the terminal to a chat client.
#
#   harness/chat/chat.sh                    # engine (the serve-pi arm) + harness/chat/cli.py
#   harness/chat/chat.sh --port 8096
#   harness/chat/chat.sh --no-serve         # attach to a server that is already running
#   CHAT_CLI='sgpt --repl naiwa --model default' harness/chat/chat.sh
#                                           # ...or any installed OpenAI-compatible CLI
#
# The server configuration is not duplicated here: this starts `harness/serve/serve-pi.sh`, the arm
# a coding agent talks to (16k context, 2 slots, MTP off), so `MODEL`, `CACHE_GB`, `CONTEXT`,
# `MIN_TOK`, `FRAC`, `MMPROJ` and `BIN` keep working exactly as they do there. MTP is left off
# (`MTP=1` turns it back on): an interactive session is where the speculator's intermittent
# `Invalid sampling probability ... NaN` shows up.
#
# Ctrl-C cancels the current *turn*, not the server: the engine is started in its own session
# (`setsid`), so it never sees the terminal's SIGINT, and the EXIT trap stops it on the way out.
#
# Env: PORT (8095), READY_TIMEOUT (900 s), OUT (~/wtmp), CHAT_CLI (an external chat command to use
#      instead of cli.py; it gets OPENAI_BASE_URL/OPENAI_API_KEY/API_BASE_URL in its environment).
set -uo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE/../.."

PORT=${PORT:-8095}
READY_TIMEOUT=${READY_TIMEOUT:-900}
SERVE=1
CLI_ARGS=()

usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; }

while [ $# -gt 0 ]; do
  case "$1" in
    --port) [ $# -ge 2 ] || { echo "--port needs a value" >&2; exit 2; }; PORT=$2; shift 2 ;;
    --no-serve) SERVE=0; shift ;;
    -h|--help) usage; exit 0 ;;
    --) shift; CLI_ARGS+=("$@"); break ;;
    *) CLI_ARGS+=("$1"); shift ;;
  esac
done

BASE="http://127.0.0.1:$PORT/v1"
LOG="${LOG:-${OUT:-$HOME/wtmp}/chat/server-$PORT.log}"
SERVE_PID=""

cleanup() {
  [ -n "$SERVE_PID" ] || return 0
  kill -0 "$SERVE_PID" 2>/dev/null || return 0
  # The engine is a session leader, so its pid is also the process-group id: signalling the group
  # takes the CUDA context and any worker threads with it instead of leaving an orphan on the GPU.
  kill -TERM "-$SERVE_PID" 2>/dev/null || kill -TERM "$SERVE_PID" 2>/dev/null
  for _ in $(seq 1 50); do
    kill -0 "$SERVE_PID" 2>/dev/null || return 0
    sleep 0.2
  done
  kill -KILL "-$SERVE_PID" 2>/dev/null || kill -KILL "$SERVE_PID" 2>/dev/null
}

# Ctrl-C belongs to the chat client in the foreground; the shell only cleans up. SIGTERM has no
# client to defer to, so it stops the engine and leaves.
trap ':' INT
trap cleanup EXIT
trap 'cleanup; exit 143' TERM

if [ "$SERVE" = "1" ]; then
  mkdir -p "$(dirname "$LOG")"
  echo "[chat] engine on $BASE (log: $LOG)" >&2
  setsid "$HERE/../serve/serve-pi.sh" "$PORT" >"$LOG" 2>&1 &
  SERVE_PID=$!

  # Weights are resident before the port is bound (serve builds the engine, then listens), so a
  # /v1/models answer means the first turn will not sit in a cold load.
  deadline=$((SECONDS + READY_TIMEOUT))
  while :; do
    if curl -sf -m 2 -o /dev/null "$BASE/models"; then
      break
    fi
    if ! kill -0 "$SERVE_PID" 2>/dev/null; then
      echo "[chat] engine exited during load; last lines of $LOG:" >&2
      tail -n 20 "$LOG" >&2
      SERVE_PID=""
      exit 1
    fi
    if [ "$SECONDS" -ge "$deadline" ]; then
      echo "[chat] engine not ready after ${READY_TIMEOUT}s; see $LOG" >&2
      exit 1
    fi
    sleep 0.5
  done
  echo "[chat] ready" >&2
fi

# An empty key satisfies SDK clients that refuse to start without one; the engine does not check it.
export OPENAI_BASE_URL="$BASE" OPENAI_API_BASE="$BASE" API_BASE_URL="$BASE" OPENAI_API_KEY=naiwa
export NAIWA_MODEL=default

# No `exec` here: the client is a child so that this shell survives to run the EXIT trap. With
# `exec` the trap went away with the shell and the engine stayed resident on the card after the
# chat ended. (Ctrl-C still reaches only the client, because the engine is in its own session.)
if [ -n "${CHAT_CLI:-}" ]; then
  # Deliberate word splitting: CHAT_CLI is a command line, not a path.
  # shellcheck disable=SC2086
  $CHAT_CLI "${CLI_ARGS[@]}"
else
  python3 "$HERE/cli.py" --base-url "$BASE" "${CLI_ARGS[@]}"
fi
exit $?
