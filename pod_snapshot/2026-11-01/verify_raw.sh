#!/usr/bin/env bash
source /workspace/activate_fasterwam_a3.sh
sha256sum /workspace/FasterWAM/checkpoints/fasterwam_release/libero/step_021700.pt > /workspace/sha_base.txt 2>&1
cd /workspace/fasterwam_a3_bridge/bridge
python3 _next/run_matrix.py --model raw --controls safety --seeds 999 --max-steps 2 --out outputs/_smoke_raw > /workspace/raw_smoke.log 2>&1
echo RAW_SMOKE_RC_$? >> /workspace/raw_smoke.log
