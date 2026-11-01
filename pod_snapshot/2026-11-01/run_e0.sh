#!/usr/bin/env bash
set -e
source /workspace/activate_fasterwam_a3.sh
cd /workspace/fasterwam_a3_bridge/bridge
python3 _next/run_matrix.py --model lora --controls C0 --seeds 11 21 22 --video --out outputs/E0_C0
python3 _next/analyze_runs.py outputs/E0_C0
echo E0_DONE
