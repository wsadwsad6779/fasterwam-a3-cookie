#!/bin/bash
set -u
LOG=/workspace/fwv1/logs/f0_continue.log
P=/workspace/fwv1/p_items
RUN=/workspace/fwv1/f0_v3_baseline
EV=/workspace/fwv1/eval_fingertip_v1
exec >> "$LOG" 2>&1
echo "F0C_WAIT $(date -u +%FT%TZ)"
while true; do
  if grep -F -q "POSTR_DONE" /workspace/fwv1/logs/post_r.log 2>/dev/null; then
    echo "F0C_POSTR_DONE $(date -u +%FT%TZ)"
    break
  fi
  if ! pgrep -f "[_]t7_post_r_chain" >/dev/null && ! pgrep -f "[_]t7_r_compare" >/dev/null \
     && ! pgrep -f "[_]t7_eval" >/dev/null && ! pgrep -f "[_]t7_train_f0.py" >/dev/null; then
    sleep 60
    if ! pgrep -f "[_]t7_post_r_chain" >/dev/null && ! pgrep -f "[_]t7_r_compare" >/dev/null \
       && ! pgrep -f "[_]t7_eval" >/dev/null && ! pgrep -f "[_]t7_train_f0.py" >/dev/null; then
      echo "F0C_PROCEED_NO_POSTR $(date -u +%FT%TZ)"
      break
    fi
  fi
  sleep 30
done
while pgrep -f "[_]t7_eval" >/dev/null || pgrep -f "[_]t7_train_f0.py" >/dev/null; do sleep 30; done
echo "F0C_BASE_ANCHOR_WAIT $(date -u +%FT%TZ)"
for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60; do
  if [ -f "$P/BASE_ANCHOR_V31.json" ] && [ -f "$P/DIM7_LOSS_GRAD_AUDIT.json" ]; then break; fi
  if [ -f "$P/BASE_ANCHOR_FAILED.json" ] || [ -f "$P/DIM7_LOSS_GRAD_FAILED.json" ]; then break; fi
  sleep 60
done
echo "F0C_BASE_ANCHOR_WAIT_DONE $(date -u +%FT%TZ)"
echo "F0C_REPORT_BEGIN $(date -u +%FT%TZ)"
/opt/venv/bin/python /workspace/_t7_final_report.py
echo "F0C_REPORT_RC=$? $(date -u +%FT%TZ)"
df -h /workspace >> "$LOG" 2>&1
CK3="$RUN/step_003000.pt"
if [ ! -f "$CK3" ]; then CK3=/root/fwv1_checkpoint_archive/f0_step_003000.pt; fi
if [ ! -f "$CK3" ]; then
  echo "F0C_STOP_NO_3K"
  exit 2
fi
echo "F0C_RESUME_3K=$CK3 $(date -u +%FT%TZ)"

for i in 1 2 3 4 5 6 7 8 9 10; do
  if [ -f "$P/F0_GRIP_SELFCHECK_3K.json" ]; then break; fi
  sleep 30
done
/opt/venv/bin/python /workspace/_f0_gate_3k.py
GATE_RC=$?
echo "F0C_GATE_ADVISORY_RC=$GATE_RC reason_recorded; continuation proceeds per user 2026-11-25T04:03:50Z $(date -u +%FT%TZ)"
echo "F0C_GATE_RELEASED_UNCONDITIONAL $(date -u +%FT%TZ)"

echo "F0C_SEG_6000_BEGIN $(date -u +%FT%TZ)"
F0_START_STEP=3000 F0_END_STEP=6000 F0_CKPT_EVERY=200 F0_RESUME="$CK3" \
  /opt/venv/bin/python /workspace/_t7_train_f0.py
RC=$?
echo "F0C_SEG_6000_TRAIN_RC=$RC $(date -u +%FT%TZ)"
if [ "$RC" -ne 0 ] || [ ! -f "$RUN/step_006000.pt" ]; then echo "F0C_STOP_6000"; exit 3; fi
T7_CKPT="$RUN/step_006000.pt" T7_STEP=6000 T7_SEEDS=7200,7201,7202,7203,7204,7205,7206,7207,7208,7209 \
  T7_OUT_SUBDIR=cmp_f0_6k T7_SCENE_MANIFEST="$P/EVAL_SCENE_MANIFEST_V3_1.json" T7_MAX_STEPS=600 T7_DENOISE=5 \
  /opt/venv/bin/python /workspace/_t7_eval.py
echo "F0C_EVAL_6000_RC=$? $(date -u +%FT%TZ)"
F0_STEP=6000 F0_SUBDIR=cmp_f0_6k F2_SUMMARY="$EV/step_006000/SUMMARY_step_6000.json" \
  /opt/venv/bin/python /workspace/_f0_milestone_report.py
/opt/venv/bin/python /workspace/_f0_storage_move.py
df -h /workspace >> "$LOG" 2>&1

echo "F0C_SEG_9000_BEGIN $(date -u +%FT%TZ)"
F0_START_STEP=6000 F0_END_STEP=9000 F0_CKPT_EVERY=200 F0_RESUME="$RUN/step_006000.pt" \
  /opt/venv/bin/python /workspace/_t7_train_f0.py
RC=$?
echo "F0C_SEG_9000_TRAIN_RC=$RC $(date -u +%FT%TZ)"
if [ "$RC" -ne 0 ] || [ ! -f "$RUN/step_009000.pt" ]; then echo "F0C_STOP_9000"; exit 4; fi
T7_CKPT="$RUN/step_009000.pt" T7_STEP=9000 T7_SEEDS=7300,7301,7302,7303,7304,7305,7306,7307,7308,7309 \
  T7_OUT_SUBDIR=cmp_f0_9k T7_SCENE_MANIFEST="$P/EVAL_SCENE_MANIFEST_V3_1.json" T7_MAX_STEPS=600 T7_DENOISE=5 \
  /opt/venv/bin/python /workspace/_t7_eval.py
echo "F0C_EVAL_9000_RC=$? $(date -u +%FT%TZ)"
F0_STEP=9000 F0_SUBDIR=cmp_f0_9k F2_SUMMARY="$EV/step_009000/SUMMARY_step_9000.json" \
  /opt/venv/bin/python /workspace/_f0_milestone_report.py
/opt/venv/bin/python /workspace/_f0_storage_move.py
df -h /workspace >> "$LOG" 2>&1

echo "F0C_SEG_12000_BEGIN $(date -u +%FT%TZ)"
F0_START_STEP=9000 F0_END_STEP=12000 F0_CKPT_EVERY=200 F0_RESUME="$RUN/step_009000.pt" \
  /opt/venv/bin/python /workspace/_t7_train_f0.py
RC=$?
echo "F0C_SEG_12000_TRAIN_RC=$RC $(date -u +%FT%TZ)"
if [ "$RC" -ne 0 ] || [ ! -f "$RUN/step_012000.pt" ]; then echo "F0C_STOP_12000"; exit 5; fi
T7_CKPT="$RUN/step_012000.pt" T7_STEP=12000 T7_SEEDS=7400,7401,7402,7403,7404,7405,7406,7407,7408,7409 \
  T7_OUT_SUBDIR=cmp_f0_12k T7_SCENE_MANIFEST="$P/EVAL_SCENE_MANIFEST_V3_1.json" T7_MAX_STEPS=600 T7_DENOISE=5 \
  /opt/venv/bin/python /workspace/_t7_eval.py
echo "F0C_EVAL_12000_RC=$? $(date -u +%FT%TZ)"
F0_STEP=12000 F0_SUBDIR=cmp_f0_12k F2_SUMMARY="$EV/step_012000/SUMMARY_step_12000.json" \
  /opt/venv/bin/python /workspace/_f0_milestone_report.py
/opt/venv/bin/python /workspace/_f0_storage_move.py
df -h /workspace >> "$LOG" 2>&1

echo "F0C_SEG_15000_BEGIN $(date -u +%FT%TZ)"
F0_START_STEP=12000 F0_END_STEP=15000 F0_CKPT_EVERY=200 F0_RESUME="$RUN/step_012000.pt" \
  /opt/venv/bin/python /workspace/_t7_train_f0.py
RC=$?
echo "F0C_SEG_15000_TRAIN_RC=$RC $(date -u +%FT%TZ)"
if [ "$RC" -ne 0 ] || [ ! -f "$RUN/step_015000.pt" ]; then echo "F0C_STOP_15000"; exit 6; fi
T7_CKPT="$RUN/step_015000.pt" T7_STEP=15000 T7_SEEDS=7500,7501,7502,7503,7504,7505,7506,7507,7508,7509 \
  T7_OUT_SUBDIR=cmp_f0_15k T7_SCENE_MANIFEST="$P/EVAL_SCENE_MANIFEST_V3_1.json" T7_MAX_STEPS=600 T7_DENOISE=5 \
  /opt/venv/bin/python /workspace/_t7_eval.py
echo "F0C_EVAL_15000_RC=$? $(date -u +%FT%TZ)"
F0_STEP=15000 F0_SUBDIR=cmp_f0_15k F2_SUMMARY="$EV/step_015000/SUMMARY_step_15000.json" \
  /opt/venv/bin/python /workspace/_f0_milestone_report.py
/opt/venv/bin/python /workspace/_f0_storage_move.py
df -h /workspace >> "$LOG" 2>&1
echo "F0C_DONE $(date -u +%FT%TZ)"