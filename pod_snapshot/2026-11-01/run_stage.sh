#!/usr/bin/env bash
set -e
C=$1
source /workspace/activate_fasterwam_a3.sh
cd /workspace/fasterwam_a3_bridge/bridge
python3 _next/run_matrix.py --model lora --controls "$C" --seeds 11 21 22 --video --out outputs/$C
python3 _next/analyze_runs.py outputs/$C
echo "$C"_DONE
