#!/bin/bash
# Independent H-sweep watcher. Waits for the 9k hold to free the GPU (GPU_FREE_FOR_MANUAL_BATCH.json),
# then runs closed-loop H=32/48/64 on F2 15k (r=H) and collects relaxed+strict in-box.
# Does NOT touch the 12k guard and does NOT train anything.
set -u
P=/workspace/fwv1/p_items
PY=/opt/venv/bin/python
LOG=/workspace/fwv1/logs/h_sweep_after_free.log
exec >> "$LOG" 2>&1
echo "HSWEEP_WAIT $(date -u +%FT%TZ)"
for i in $(seq 1 720); do [ -f "$P/GPU_FREE_FOR_MANUAL_BATCH.json" ] && break; sleep 20; done
echo "HSWEEP_GPU_FREE_SEEN $(date -u +%FT%TZ)"
while pgrep -f "[_]t7_train_f0.py" >/dev/null || pgrep -f "[_]t7_eval" >/dev/null; do sleep 10; done
F2=/workspace/fwv1/train_runs/t7_21k_v3run/step_015000.pt
if [ ! -f "$F2" ]; then echo "HSWEEP_NO_F2"; exit 2; fi
echo "HSWEEP_CKPT=$F2 $(date -u +%FT%TZ)"
for H in 32 48 64; do
  echo "HSWEEP_RUN H=$H $(date -u +%FT%TZ)"
  T7_H=$H T7_CKPT="$F2" T7_STEP=h$H \
  T7_SEEDS=7500,7501,7502,7503,7504,7505,7506,7507,7508,7509 \
  T7_OUT_SUBDIR=h_eval_h$H T7_SCENE_MANIFEST="$P/EVAL_SCENE_MANIFEST_V3_1.json" \
  T7_MAX_STEPS=600 T7_DENOISE=5 \
    timeout 3600 $PY /workspace/_h_eval.py
  echo "HSWEEP_RC H=$H $? $(date -u +%FT%TZ)"
done
$PY /workspace/_h_sweep_collect.py
echo "HSWEEP_DONE $(date -u +%FT%TZ)"
