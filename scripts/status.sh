#!/usr/bin/env bash
# One-screen status: is the runner alive, disk, queue, latest history, current run's last epochs.
cd "$(dirname "$0")/.."
if [ -f runs/queue.pid ] && kill -0 "$(cat runs/queue.pid)" 2>/dev/null; then echo "runner: RUNNING (pid $(cat runs/queue.pid))"; else echo "runner: not running  -> bash scripts/start_queue.sh"; fi
echo "disk free: $(df -h . | tail -1 | awk '{print $4}')"
echo "--- queue (configs/queue.txt) ---"
grep -vE '^\s*(#|$)' configs/queue.txt 2>/dev/null | while read -r line; do
  n=${line%% *}; s="todo"; [ -f "runs/$n/DONE" ] && s="DONE"; [ -f "runs/$n/FAILS" ] && s="failed x$(cat runs/$n/FAILS)"
  [ -f "runs/$n/weights/last.pt" ] && [ "$s" = todo ] && s="in progress/resumable"
  echo "  [$s] $line"
done
echo "--- runs/history.log (last 10) ---"; tail -10 runs/history.log 2>/dev/null
echo "--- runs/queue.log (last 5) ---"; tail -5 runs/queue.log 2>/dev/null | cut -c1-200
latest=$(ls -td runs/*/results.csv 2>/dev/null | head -1)
if [ -n "$latest" ]; then echo "--- $latest (last 3 epochs: epoch, time, mAP50, mAP50-95) ---"
  python3 - "$latest" <<'EOF'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
rows = [{k.strip(): v.strip() for k, v in r.items()} for r in rows]
for r in rows[-3:]:
    print(r.get("epoch"), r.get("time"), r.get("metrics/mAP50(B)"), r.get("metrics/mAP50-95(B)"))
EOF
fi
