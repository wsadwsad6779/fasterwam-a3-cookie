#!/usr/bin/env bash
set -e
source /workspace/activate_fasterwam_a3.sh
cd /workspace/fasterwam_a3_bridge/bridge
python3 -m py_compile _next/*.py
python3 _next/selftest.py
python3 _next/run_matrix.py --model raw  --controls safety --seeds 999 --max-steps 2 --out outputs/_smoke_raw
python3 _next/run_matrix.py --model lora --controls safety --seeds 999 --max-steps 2 --out outputs/_smoke_lora
python3 _next/b0_probe.py --out outputs/b0
python3 _next/b0_probe.py --model raw  --out outputs/b0_raw
python3 _next/b0_probe.py --model lora --out outputs/b0_lora
echo B0_DONE
