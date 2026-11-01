#!/usr/bin/env bash
set -u
source /workspace/activate_fasterwam_a3.sh
python3 -u /workspace/_r4_stage/keys_eval.py
echo "avail=$(df -B1 --output=avail /workspace | tail -1 | tr -d " ")"
ls /workspace/a3_train_runs/a3_lora_r4/checkpoints/light/