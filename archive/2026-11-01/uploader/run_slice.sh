#!/bin/bash
# usage: bash run_slice.sh TAG IDLE_SECS TRIALS <python args...>
# marks/logs land under /workspace/_r6_stage/out so the Jupyter contents API can read them
set -u
TAG=$1; IDLE=$2; TRIALS=$3; shift 3
LOGD=/root/hfstage
OUTD=/workspace/_r6_stage/out
MARK=$OUTD/run_$TAG.mark
RUNLOG=$OUTD/run_$TAG.log
: > "$RUNLOG"; : > "$MARK"
cd "$LOGD" || { echo "cd failed" >> "$MARK"; exit 9; }
for try in $(seq 1 "$TRIALS"); do
  echo "try=$try start $(date -u +%FT%TZ)" >> "$MARK"
  python3 -u hf_archive.py "$@" >> "$RUNLOG" 2>&1 &
  PID=$!
  while kill -0 "$PID" 2>/dev/null; do
    sleep 20
    L=$(ls -t "$LOGD"/xet/logs/*.log 2>/dev/null | head -1)
    NOW=$(date +%s)
    MT=$(stat -c %Y "$L" 2>/dev/null || echo "$NOW")
    AGE=$(( NOW - MT ))
    echo "watch try=$try pid=$PID xet_idle=${AGE}s $(date -u +%FT%H%M%SZ)" >> "$MARK"
    if [ "$AGE" -gt "$IDLE" ]; then
      echo "try=$try IDLE=${AGE}s KILL=$PID $(date -u +%FT%TZ)" >> "$MARK"
      kill -9 "$PID" 2>/dev/null
      pkill -9 -f hf_archive.py 2>/dev/null
      break
    fi
  done
  wait "$PID" 2>/dev/null
  rc=$?
  echo "try=$try rc=$rc end $(date -u +%FT%TZ)" >> "$MARK"
  if [ "$rc" -eq 0 ]; then
    echo "try=$try COMPLETED" >> "$MARK"
    break
  fi
  sleep 15
done
echo "FIN $TAG $(date -u +%FT%TZ)" >> "$MARK"
