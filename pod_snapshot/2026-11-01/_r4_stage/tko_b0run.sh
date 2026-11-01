#!/usr/bin/env bash
set -u
source /workspace/activate_fasterwam_a3.sh
cd /workspace/fasterwam_a3_bridge/bridge/_next || exit 1
{
  date -u +%FT%TZ
  echo == B0 seed22 read-only mask replay
  python3 -u /workspace/_r4_stage/tko_mask.py \
    --npz /workspace/a3_eval/r4_b0_c0_paired/lora_B0_22.npz \
    --seed 22 \
    --out /workspace/_r4_stage/tko_mask_B0_22.npz \
    --movers 0 1 2 3 4 5 6 7 8 9
  echo mask_rc=$?
  ls -la /workspace/_r4_stage/tko_mask_B0_22.npz
  sha256sum /workspace/_r4_stage/tko_mask_B0_22.npz
  date -u +%FT%TZ
} > /workspace/_r4_stage/tko_b0.log 2>&1
echo __B0MASK_DONE__ >> /workspace/_r4_stage/tko_b0.log