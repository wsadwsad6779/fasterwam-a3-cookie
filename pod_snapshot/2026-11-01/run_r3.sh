#!/usr/bin/env bash
set -e
source /workspace/activate_fasterwam_a3.sh
export FW_SKIP_STATE=1
cd /workspace/FasterWAM
accelerate launch --num_processes 1 scripts/train.py data=a3_2cam task=a3_fasterwam batch_size=6 max_steps=3000 learning_rate=1e-4 log_every=20 save_every=3000 eval_every=0 output_dir=/workspace/a3_train_runs/a3_lora_r3 resume=/workspace/a3_train_runs/a3_lora_r2/step_003000_merged.pt
