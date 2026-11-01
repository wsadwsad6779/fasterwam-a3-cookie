#!/usr/bin/env bash
set -u
source /workspace/activate_fasterwam_a3.sh
export FASTERWAM_A3_INSTRUCTION='transfer 10 cookies into target box'
cd /workspace/fasterwam_a3_bridge/bridge/_next || exit 1
O=/workspace/a3_eval/r4_c0_seed21_video
L=/workspace/a3_eval/r4_c0_seed21_video.log
mkdir -p $O
{
  date -u +%FT%TZ
  echo == writer smoke
  python3 /workspace/_r4_stage/writer_smoke.py
  echo == r4 C0 seed21 video run start
  python3 run_matrix.py \
    --model lora \
    --ckpt /workspace/a3_train_runs/a3_lora_r3/step_003000_merged.pt \
    --adapter /workspace/a3_train_runs/a3_lora_r4/checkpoints/light/step_004000/adapter.pt \
    --stats /workspace/a3_data/a3_fw/dataset_stats.json \
    --gripper-mode pm1 \
    --controls C0 \
    --seeds 21 \
    --replan 10 --max-steps 600 \
    --position-noise 0 --yaw-noise 0 --batch-offset 0 \
    --video \
    --out $O
  echo eval_exit=$?
  ls -la $O
  df -h /workspace
  date -u +%FT%TZ
} > $L 2>&1
echo __R4VID_DONE__ >> $L