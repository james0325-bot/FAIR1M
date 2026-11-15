#!/usr/bin/env bash
# Idempotent data preparation: wait for / finish the download, verify it, convert to 5-class OBB.
# Safe to run any number of times; does nothing if everything is already in place.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
RAW=datasets/raw/fair1m_hf
DST=datasets/processed/fair1m5
log() { echo "[prepare $(date '+%F %T')] $*"; }

# 1. let a running download finish (if any)
while pgrep -f "huggingface_hub import snapshot_download" >/dev/null; do sleep 20; done

# 2. verify / resume (download is resumable)
count() { find "$RAW/data/$1" -name "*.$2" 2>/dev/null | wc -l | tr -d ' '; }
for try in 1 2 3 4; do
  T=$(count images tif); X=$(count labelXmls xml)
  log "check $try: tif=$T xml=$X (want 1732/1732)"
  [ "$T" = 1732 ] && [ "$X" = 1732 ] && break
  log "incomplete, (re)starting download"
  $PY -I -c "from huggingface_hub import snapshot_download as s; s(repo_id='blanchon/FAIR1M', repo_type='dataset', local_dir='$RAW', max_workers=8)" || true
done
T=$(count images tif); X=$(count labelXmls xml)
[ "$T" = 1732 ] && [ "$X" = 1732 ] || { log "FAILED: download incomplete tif=$T xml=$X"; exit 1; }

# 3. convert once
if [ -f "$DST/.complete" ] && [ -f configs/fair1m5.yaml ]; then
  log "processed data already present"
else
  log "converting"
  $PY -I scripts/convert_fair1m.py --force || { log "FAILED: convert"; exit 1; }
  touch "$DST/.complete"
  du -sh "$DST"
fi
log "data ready"
