#!/usr/bin/env bash
set -u
source /workspace/activate_fasterwam_a3.sh
export FASTERWAM_A3_INSTRUCTION='transfer 10 cookies into target box'
cd /workspace/fasterwam_a3_bridge/bridge/_next || exit 1
{
  date -u +%FT%TZ
  echo '== r4 C0 eval start (seeds 11 21 22, C0, 600 steps, replan 10)'
  python3 run_matrix.py \
    --model lora \
    --ckpt /workspace/a3_train_runs/a3_lora_r3/step_003000_merged.pt \
    --adapter /workspace/a3_train_runs/a3_lora_r4/checkpoints/light/step_004000/adapter.pt \
    --stats /workspace/a3_data/a3_fw/dataset_stats.json \
    --gripper-mode pm1 \
    --controls C0 \
    --seeds 11 21 22 \
    --replan 10 --max-steps 600 \
    --position-noise 0 --yaw-noise 0 --batch-offset 0 \
    --out /workspace/a3_eval/r4_c0_frozen
  echo "eval_exit=$?"
  date -u +%FT%TZ
} > /workspace/a3_eval/r4_c0_frozen.log 2>&1
echo __R4EVAL_DONE__ >> /workspace/a3_eval/r4_c0_frozen.log
