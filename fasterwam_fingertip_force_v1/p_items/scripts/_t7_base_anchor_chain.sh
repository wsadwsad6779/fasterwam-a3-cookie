#!/bin/bash
set -u
LOG=/workspace/fwv1/logs/base_anchor.log
P=/workspace/fwv1/p_items
exec >> "$LOG" 2>&1
echo "BASE_WAIT $(date -u +%FT%TZ)"
while true; do
  if grep -F -q "POSTR_DONE" /workspace/fwv1/logs/post_r.log 2>/dev/null; then
    echo "BASE_POSTR_DONE $(date -u +%FT%TZ)"
    break
  fi
  if ! pgrep -f "[_]t7_post_r_chain" >/dev/null && ! pgrep -f "[_]t7_r_compare" >/dev/null \
     && ! pgrep -f "[_]t7_eval" >/dev/null; then
    sleep 60
    if ! pgrep -f "[_]t7_post_r_chain" >/dev/null && ! pgrep -f "[_]t7_r_compare" >/dev/null \
       && ! pgrep -f "[_]t7_eval" >/dev/null; then
      echo "BASE_PROCEED_NO_POSTR $(date -u +%FT%TZ)"
      break
    fi
  fi
  sleep 30
done
while pgrep -f "[_]t7_eval" >/dev/null; do sleep 30; done
echo "DIM7_AUDIT_BEGIN $(date -u +%FT%TZ)"
/opt/venv/bin/python /workspace/_dim7_loss_audit.py
echo "DIM7_AUDIT_RC=$? $(date -u +%FT%TZ)"
echo "BASE_EVAL_BEGIN $(date -u +%FT%TZ)"
T7_STEP=base T7_SEEDS=7100,7101,7102,7103,7104,7105,7106,7107,7108,7109 \
  T7_MAX_STEPS=600 T7_DENOISE=5 T7_OUT_SUBDIR=base_v31_thinpads \
  T7_SCENE_MANIFEST="$P/EVAL_SCENE_MANIFEST_V3_1.json" \
  /opt/venv/bin/python /workspace/_t7_eval.py
RC=$?
echo "BASE_EVAL_RC=$RC $(date -u +%FT%TZ)"
if [ "$RC" -ne 0 ]; then
  echo "BASE_FAILED" > "$P/BASE_ANCHOR_FAILED.json"
  echo "BASE_ABORT $(date -u +%FT%TZ)"
  exit 2
fi
ZIP_SUBDIR=base_v31_thinpads ZIP_LABEL=base_v31 /opt/venv/bin/python /workspace/_zip_eval_videos.py
echo "BASE_ZIP_RC=$? $(date -u +%FT%TZ)"
/opt/venv/bin/python /workspace/_base_anchor_report.py
echo "BASE_REPORT_RC=$? $(date -u +%FT%TZ)"
echo "BASE_DONE $(date -u +%FT%TZ)"