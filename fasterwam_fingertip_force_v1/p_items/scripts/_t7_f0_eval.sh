#!/bin/bash
set -u
LOG=/workspace/fwv1/logs/f0_eval.log
exec >> "$LOG" 2>&1
echo "F0_EVAL_WAIT $(date -u +%FT%TZ)"
while pgrep -f "[_]t7_train_f0.py" >/dev/null; do
  sleep 30
done
echo "F0_TRAIN_EXITED $(date -u +%FT%TZ)"
F0CK=/workspace/fwv1/f0_v3_baseline/step_003000.pt
if [ ! -f "$F0CK" ]; then
  echo "F0_EVAL_MISSING $F0CK"
  exit 1
fi
SEEDS=7100,7101,7102,7103,7104,7105,7106,7107,7108,7109
T7_CKPT="$F0CK" T7_STEP=3000 T7_SEEDS="$SEEDS" T7_OUT_SUBDIR=cmp_f0_3k \
  T7_SCENE_MANIFEST=/workspace/fwv1/p_items/EVAL_SCENE_MANIFEST_V3_1.json \
  T7_MAX_STEPS=600 T7_DENOISE=5 /opt/venv/bin/python /workspace/_t7_eval.py
echo "F0_EVAL_RC=$? $(date -u +%FT%TZ)"
F2CK=/workspace/fwv1/train_runs/t7_21k_v3run/step_003000.pt
if [ ! -f "$F2CK" ]; then
  F2CK=/root/fwv1_checkpoint_archive/step_003000.pt
fi
echo "F2_3K_CKPT=$F2CK"
T7_CKPT="$F2CK" T7_STEP=3000 T7_SEEDS="$SEEDS" T7_OUT_SUBDIR=cmp_f2_3k \
  T7_SCENE_MANIFEST=/workspace/fwv1/p_items/EVAL_SCENE_MANIFEST_V3_1.json \
  T7_MAX_STEPS=600 T7_DENOISE=5 /opt/venv/bin/python /workspace/_t7_eval.py
echo "F2_3K_EVAL_RC=$? $(date -u +%FT%TZ)"
/opt/venv/bin/python /workspace/_t7_cmp_3k.py
echo "F0_EVAL_DONE $(date -u +%FT%TZ)"