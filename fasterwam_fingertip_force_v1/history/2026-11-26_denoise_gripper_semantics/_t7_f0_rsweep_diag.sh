#!/bin/bash
# F0 15k r-sweep (r=10/16/32) x seeds 7500-7509, SAME diag evaluator v2 (strict+relaxed+trajectory).
# H=32, K=5, V3.1 frozen, 600 steps, r=min(r,32). No training. Reuses one evaluator across all r.
set -u
PY=/opt/venv/bin/python
P=/workspace/fwv1/p_items
EV=/workspace/fwv1/eval_fingertip_v1
CK=/workspace/fwv1/f0_v3_baseline/step_015000.pt
LOG=/workspace/fwv1/logs/f0_rsweep_diag.log
SEEDS=7500,7501,7502,7503,7504,7505,7506,7507,7508,7509
exec >> "$LOG" 2>&1
echo "RS_BEGIN $(date -u +%FT%TZ) ckpt=$CK"
[ -f "$CK" ] || { echo "RS_NO_CKPT"; exit 2; }
for R in 10 16 32; do
  echo "RS_RUN r=$R $(date -u +%FT%TZ)"
  T7_CKPT="$CK" T7_STEP=f0r$R T7_REPLAN=$R T7_SEEDS=$SEEDS \
  T7_OUT_SUBDIR=f0_15k_r$R T7_SCENE_MANIFEST="$P/EVAL_SCENE_MANIFEST_V3_1.json" \
  T7_MAX_STEPS=600 T7_DENOISE=5 \
    $PY /workspace/_t7_eval_diag_v2.py
  echo "RS_RC r=$R $? $(date -u +%FT%TZ)"
done
$PY /workspace/_f0_rsweep_collect.py
echo "RS_DONE $(date -u +%FT%TZ)"
