#!/bin/bash
# Waits for the 21k chain to finish, then runs degradation analysis + S5 final independent test.
LOG=/workspace/fwv1/logs/t7_finalizer.log
SUM=/workspace/fwv1/eval_fingertip_v1/step_021000/SUMMARY_step_21000.json
OUT=/workspace/fwv1/p_items/FINAL_PIPELINE_DONE.json
exec >> "$LOG" 2>&1
echo "[finalizer] start $(date -u +%Y-%m-%dT%H:%M:%SZ); waiting for $SUM"
while [ ! -f "$SUM" ]; do
  sleep 60
done
while pgrep -f "_t7_train.py" > /dev/null; do
  sleep 30
done
echo "[finalizer] 21k summary present $(date -u +%Y-%m-%dT%H:%M:%SZ)"
/opt/venv/bin/python /workspace/_t7_degradation_analysis.py
echo "[finalizer] degradation rc=$?"
/opt/venv/bin/python /workspace/_t7_final_test.py
RC=$?
echo "[finalizer] final test rc=$RC $(date -u +%Y-%m-%dT%H:%M:%SZ)"
python3 - "$RC" <<'PY'
import json, sys, time
json.dump({"worker": "task7", "ts": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
           "final_test_rc": int(sys.argv[1]), "degradation_report": "/workspace/fwv1/p_items/DEGRADATION_REPORT.json",
           "final_report": "/workspace/fwv1/p_items/FINAL_TEST_REPORT.md",
           "force_contribution": "/workspace/fwv1/p_items/FORCE_CONTRIBUTION.json",
           "video_manifest": "/workspace/fwv1/p_items/VIDEO_MANIFEST_final.json",
           "best_selection": "/workspace/fwv1/p_items/BEST_CHECKPOINT_SELECTION.json"},
          open('/workspace/fwv1/p_items/FINAL_PIPELINE_DONE.json', 'w'), indent=1)
PY
echo "[finalizer] DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"