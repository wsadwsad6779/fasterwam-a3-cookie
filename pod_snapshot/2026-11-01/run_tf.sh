#!/usr/bin/env bash
set -e
source /workspace/activate_fasterwam_a3.sh
cd /workspace/fasterwam_a3_bridge/bridge
python3 _next/run_teacher_forcing.py --model lora --seeds 11 --max-steps 600 --query-stride 20 --gripper-modes pm1 --out outputs/tf_lora
echo TF_LORA_DONE
