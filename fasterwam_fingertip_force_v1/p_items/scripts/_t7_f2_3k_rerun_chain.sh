#!/bin/bash
set -u
LOG=/workspace/fwv1/logs/f2_3k_rerun.log
P=/workspace/fwv1/p_items
exec >> "$LOG" 2>&1
echo "F2R_BEGIN $(date -u +%FT%TZ)"
CK=/root/fwv1_checkpoint_archive/step_003000.pt
if [ ! -f "$CK" ]; then
  echo "F2R_NO_CKPT $CK"
  exit 2
fi
sha256sum "$CK"
T7_CKPT="$CK" T7_STEP=3000 T7_SEEDS=7100,7101,7102,7103,7104,7105,7106,7107,7108,7109 \
  T7_OUT_SUBDIR=cmp_f2_3k T7_SCENE_MANIFEST="$P/EVAL_SCENE_MANIFEST_V3_1.json" \
  T7_MAX_STEPS=600 T7_DENOISE=5 \
  /opt/venv/bin/python /workspace/_t7_eval.py
echo "F2R_EVAL_RC=$? $(date -u +%FT%TZ)"
/opt/venv/bin/python /workspace/_t7_cmp_3k.py
echo "F2R_CMP_RC=$? $(date -u +%FT%TZ)"
ZIP_SUBDIR=cmp_f2_3k ZIP_LABEL=f2_step_3000 /opt/venv/bin/python /workspace/_zip_eval_videos.py
echo "F2R_ZIP_RC=$? $(date -u +%FT%TZ)"
/opt/venv/bin/python /workspace/_f2_eval_done_evidence.py
echo "F2R_EVIDENCE_RC=$? $(date -u +%FT%TZ)"
/opt/venv/bin/python /workspace/_base_anchor_report.py
echo "F2R_BASE_RC=$? $(date -u +%FT%TZ)"
echo "F2R_DONE $(date -u +%FT%TZ)"
tmux new-window -t fwv1 -n r_compare 'bash /workspace/_t7_r_compare_chain.sh'
echo "F2R_R_SWEEP_STARTED $(date -u +%FT%TZ)"