#!/usr/bin/env bash
set -e
source /workspace/activate_fasterwam_a3.sh
cd /workspace/FasterWAM
STEPS=${1:-3000}
OUT=${2:-/workspace/a3_train_runs/a3_lora_r2}
accelerate launch --num_processes 1 scripts/train.py data=a3_2cam task=a3_fasterwam \
  batch_size=6 max_steps=$STEPS learning_rate=1e-4 log_every=20 \
  save_every=1500 eval_every=0 \
  output_dir=$OUT \
  resume=/workspace/a3_train_runs/a3_lora_r1/step_002000_merged.pt
