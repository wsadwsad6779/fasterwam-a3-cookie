#!/usr/bin/env bash
# One-off run of stage 0b alone: the dataset->trainer force seam, no model, no GPU.
set -u
STAGE=/workspace/_v1_stage
FW=$STAGE/fw
LED=$STAGE/rehearsal65
PY=/opt/venv/bin/python
mkdir -p "$LED"
cd "$FW" || exit 9
export PYTHONPATH=$FW/src
export FASTERWAM_A3_INSTRUCTION="pick up the cookie and place it in the box"
export TOKENIZERS_PARALLELISM=false
CUDA_VISIBLE_DEVICES="" $PY -u scripts/verify_dataset_force.py \
  data=a3_2cam_force task=a3_force_40k batch_size=6 max_steps=40000 \
  learning_rate=1e-4 log_every=1 eval_every=0 \
  output_dir="$LED/_dsforce" +report="$LED/dataset_force_verify.json" \
  > "$LED"/dataset_force.log 2>&1
echo "@@DSFORCE_RC@@ $?"
tail -6 "$LED"/dataset_force.log
