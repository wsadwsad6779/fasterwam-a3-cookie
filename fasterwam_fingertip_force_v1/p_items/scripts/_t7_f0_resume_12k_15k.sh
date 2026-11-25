#!/bin/bash
# F0 resume gate (user-authorized): 9k -> 12k (save+eval) -> 15k (stop/eval/freeze).
# Idempotent: skips a segment whose final ckpt + eval SUMMARY already exist. Hard stop 15000.
# Uses the deployed trainer/eval; eval is fail-closed. Does NOT touch F2. No formula change.
set -u
LOG=/workspace/fwv1/logs/f0_resume_12k_15k.log
P=/workspace/fwv1/p_items
RUN=/workspace/fwv1/f0_v3_baseline
EV=/workspace/fwv1/eval_fingertip_v1
PY=/opt/venv/bin/python
exec >> "$LOG" 2>&1
echo "RESUME_BEGIN $(date -u +%FT%TZ)"
CK9="$RUN/step_009000.pt"; [ -f "$CK9" ] || CK9=/root/fwv1_checkpoint_archive/f0_step_009000.pt
[ -f "$CK9" ] || { echo "RESUME_NO_9K"; echo '{"status":"FAIL","reason":"no step_009000.pt"}' > "$P/F0_RESUME_FAIL.json"; exit 2; }
echo "RESUME_FROM=$CK9 $(date -u +%FT%TZ)"

# ---- 9k -> 12k ----
if [ -f "$RUN/step_012000.pt" ] && [ -f "$EV/cmp_f0_12k/SUMMARY_step_12000.json" ]; then
  echo "RESUME_SKIP_12000_DONE"
else
  echo "RESUME_SEG_12000_BEGIN $(date -u +%FT%TZ)"
  F0_START_STEP=9000 F0_END_STEP=12000 F0_CKPT_EVERY=200 F0_RESUME="$CK9" $PY /workspace/_t7_train_f0.py
  RC=$?; echo "RESUME_SEG_12000_RC=$RC $(date -u +%FT%TZ)"
  if [ "$RC" -ne 0 ] || [ ! -f "$RUN/step_012000.pt" ]; then echo "RESUME_STOP_12000"; echo '{"status":"FAIL","seg":"12000"}' > "$P/F0_RESUME_FAIL.json"; exit 3; fi
  T7_CKPT="$RUN/step_012000.pt" T7_STEP=12000 \
  T7_SEEDS=7400,7401,7402,7403,7404,7405,7406,7407,7408,7409 \
  T7_OUT_SUBDIR=cmp_f0_12k T7_SCENE_MANIFEST="$P/EVAL_SCENE_MANIFEST_V3_1.json" T7_MAX_STEPS=600 T7_DENOISE=5 \
    $PY /workspace/_t7_eval.py
  echo "RESUME_EVAL_12000_RC=$? $(date -u +%FT%TZ)"
  F0_STEP=12000 F0_SUBDIR=cmp_f0_12k F2_SUMMARY="$EV/step_012000/SUMMARY_step_12000.json" $PY /workspace/_f0_milestone_report.py
  $PY /workspace/_f0_storage_move.py
fi

# ---- 12k -> 15k (hard stop) ----
if [ -f "$RUN/step_015000.pt" ] && [ -f "$EV/cmp_f0_15k/SUMMARY_step_15000.json" ]; then
  echo "RESUME_SKIP_15000_DONE"
else
  echo "RESUME_SEG_15000_BEGIN $(date -u +%FT%TZ)"
  F0_START_STEP=12000 F0_END_STEP=15000 F0_CKPT_EVERY=200 F0_RESUME="$RUN/step_012000.pt" $PY /workspace/_t7_train_f0.py
  RC=$?; echo "RESUME_SEG_15000_RC=$RC $(date -u +%FT%TZ)"
  if [ "$RC" -ne 0 ] || [ ! -f "$RUN/step_015000.pt" ]; then echo "RESUME_STOP_15000"; echo '{"status":"FAIL","seg":"15000"}' > "$P/F0_RESUME_FAIL.json"; exit 4; fi
  T7_CKPT="$RUN/step_015000.pt" T7_STEP=15000 \
  T7_SEEDS=7500,7501,7502,7503,7504,7505,7506,7507,7508,7509 \
  T7_OUT_SUBDIR=cmp_f0_15k T7_SCENE_MANIFEST="$P/EVAL_SCENE_MANIFEST_V3_1.json" T7_MAX_STEPS=600 T7_DENOISE=5 \
    $PY /workspace/_t7_eval.py
  echo "RESUME_EVAL_15000_RC=$? $(date -u +%FT%TZ)"
  F0_STEP=15000 F0_SUBDIR=cmp_f0_15k F2_SUMMARY="$EV/step_015000/SUMMARY_step_15000.json" $PY /workspace/_f0_milestone_report.py
  $PY /workspace/_f0_storage_move.py
fi

# ---- freeze at 15k (hard stop; NO 15k->21k) ----
SUM=""
[ -f "$EV/cmp_f0_15k/SUMMARY_step_15000.json" ] && SUM=$(sha256sum "$EV/cmp_f0_15k/SUMMARY_step_15000.json" | cut -d' ' -f1)
CK15=""
[ -f "$RUN/step_015000.pt" ] && CK15=$(sha256sum "$RUN/step_015000.pt" | cut -d' ' -f1)
$PY -c "import json,time;json.dump({'status':'F0_FROZEN_15K','ts':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'checkpoint_sha256':'$CK15','summary15k_sha256':'$SUM','hard_stop':15000,'note':'15k terminal; no 15k->21k'},open('/workspace/fwv1/p_items/F0_FROZEN_15K.json','w'),indent=1)"
echo "RESUME_DONE $(date -u +%FT%TZ)"
