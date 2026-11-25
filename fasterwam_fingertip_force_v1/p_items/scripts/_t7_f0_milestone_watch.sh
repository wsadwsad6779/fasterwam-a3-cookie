#!/bin/bash
set -u
LOG=/workspace/fwv1/logs/f0_milestone_watch.log
P=/workspace/fwv1/p_items
EV=/workspace/fwv1/eval_fingertip_v1
exec >> "$LOG" 2>&1
echo "F0WATCH_START $(date -u +%FT%TZ)"
for spec in "6000 cmp_f0_6k" "9000 cmp_f0_9k" "12000 cmp_f0_12k" "15000 cmp_f0_15k"; do
  set -- $spec
  STEP=$1
  SUB=$2
  echo "F0WATCH_WAIT step=$STEP sub=$SUB $(date -u +%FT%TZ)"
  while [ ! -f "$EV/$SUB/SUMMARY_step_$STEP.json" ]; do
    sleep 30
  done
  echo "F0WATCH_FOUND step=$STEP $(date -u +%FT%TZ)"
  ZIP_SUBDIR=$SUB ZIP_LABEL=f0_step_$STEP /opt/venv/bin/python /workspace/_zip_eval_videos.py
  echo "F0WATCH_ZIP_RC=$? step=$STEP $(date -u +%FT%TZ)"
  F0_STEP=$STEP F0_SUBDIR=$SUB /opt/venv/bin/python /workspace/_f0_milestone_evidence.py
  echo "F0WATCH_EVIDENCE_RC=$? step=$STEP $(date -u +%FT%TZ)"
  sleep 5
done
echo "F0WATCH_DONE $(date -u +%FT%TZ)"