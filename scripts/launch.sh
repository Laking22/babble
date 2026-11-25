#!/bin/bash
# Babble launcher: start the app if it isn't running, then open it in the browser.
APP_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PORT=7895
URL="http://127.0.0.1:$PORT"
LOG="$HOME/Library/Logs/Babble.log"
up() { curl -s -o /dev/null -m 2 "$URL"; }
if ! up; then
  cd "$APP_DIR" || exit 1
  unset NARRATOR_HOME   # always your real data: ~/Library/Application Support/Babble
  BABBLE_MOVE_DATA=1 nohup "$APP_DIR/.venv/bin/python" run.py --port "$PORT" >> "$LOG" 2>&1 &
  for i in $(seq 1 90); do up && break; sleep 1; done
fi
if up; then
  # the engine answers only with this run's key (narrator/guard.py): the page takes it from the link
  TOKEN="$(cat "$HOME/Library/Application Support/Babble/engine-token" 2>/dev/null \
    || cat "$HOME/Library/Application Support/Narrator Studio/engine-token" 2>/dev/null)"   # (not moved yet)
  open "$URL/app/#token=$TOKEN"
else
  osascript -e 'display alert "Babble didn’t start" message "See ~/Library/Logs/Babble.log for details."'
fi
