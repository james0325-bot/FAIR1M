#!/usr/bin/env bash
# Training queue runner. Reads configs/queue.txt, runs each not-yet-finished entry in order.
# - Re-reads the file before every job, so entries can be added/edited while it runs.
# - A finished job gets runs/<name>/DONE and is skipped; an interrupted job RESUMES from last.pt.
# - A job that fails 3 times is skipped (runs/<name>/FAILS holds the count).
# - Only one runner at a time (runs/queue.pid).
# Start it detached from the terminal with:  bash scripts/start_queue.sh
set -u
cd "$(dirname "$0")/.."
export PYTORCH_ENABLE_MPS_FALLBACK=1
PY=.venv/bin/python
PIDFILE=runs/queue.pid
mkdir -p runs
log() { echo "[queue $(date '+%F %T')] $*"; }

if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then log "already running (pid $(cat "$PIDFILE"))"; exit 0; fi
echo $$ > "$PIDFILE"
trap 'rm -f "$PIDFILE"' EXIT

bash scripts/prepare_data.sh || { log "data preparation failed"; exit 1; }

while true; do
  next=""
  while IFS= read -r line; do
    case "$line" in ''|\#*) continue;; esac
    n=${line%% *}
    [ -f "runs/$n/DONE" ] && continue
    f=$(cat "runs/$n/FAILS" 2>/dev/null || echo 0)
    [ "$f" -ge 3 ] && continue
    next="$line"; break
  done < configs/queue.txt
  if [ -z "$next" ]; then log "queue empty, exiting"; break; fi
  name=${next%% *}; args=${next#"$name"}
  fails=$(cat "runs/$name/FAILS" 2>/dev/null || echo 0)
  log "start $name"
  # shellcheck disable=SC2086
  if $PY scripts/train.py "$name" $args; then
    mkdir -p "runs/$name"; touch "runs/$name/DONE"; log "done $name"
  else
    mkdir -p "runs/$name"; echo $((fails + 1)) > "runs/$name/FAILS"; log "FAILED $name (fails=$((fails + 1)))"; sleep 30
  fi
done
