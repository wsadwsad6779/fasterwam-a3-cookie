#!/usr/bin/env bash
set -u
source /workspace/activate_fasterwam_a3.sh
cd /workspace/fasterwam_a3_bridge/bridge/_next || exit 1
{
  date -u +%FT%TZ
  echo == sha of tko_mask.py on pod
  sha256sum /workspace/_r4_stage/tko_mask.py
  wc -c /workspace/_r4_stage/tko_mask.py
  echo == seed21 npz vs frozen
  sha256sum /workspace/a3_eval/r4_c0_frozen/lora_C0_21.npz /workspace/a3_eval/r4_c0_seed21_video/lora_C0_21.npz
  echo == replay mask seed22
  python3 -u /workspace/_r4_stage/tko_mask.py --npz /workspace/a3_eval/r4_c0_seed22_video/lora_C0_22.npz --seed 22 --out /workspace/_r4_stage/tko_mask_seed22.npz --movers 0 1 2 3 4 5
  echo mask_rc=$?
  date -u +%FT%TZ
} > /workspace/_r4_stage/tko_mask.log 2>&1
echo __MASK_DONE__ >> /workspace/_r4_stage/tko_mask.log
