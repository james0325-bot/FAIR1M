#!/usr/bin/env bash
# Start the training queue fully detached from this terminal (new session + caffeinate).
# Closing the terminal / the Claude session does NOT stop it. Safe to run again: a second runner refuses to start.
#   bash scripts/start_queue.sh      # start (or no-op if already running)
#   bash scripts/status.sh           # see what it is doing
#   kill $(cat runs/queue.pid)       # stop the runner (the current epoch's last.pt is kept; rerun to resume)
cd "$(dirname "$0")/.."
mkdir -p runs
python3 - <<'EOF'
import subprocess
subprocess.Popen(
    ["caffeinate", "-dimsu", "bash", "scripts/run_queue.sh"],
    stdout=open("runs/queue.log", "ab"), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
    start_new_session=True,
)
EOF
sleep 2
if [ -f runs/queue.pid ] && kill -0 "$(cat runs/queue.pid)" 2>/dev/null; then echo "queue runner started (pid $(cat runs/queue.pid)); log: runs/queue.log"; else echo "queue runner not running; see runs/queue.log"; fi
