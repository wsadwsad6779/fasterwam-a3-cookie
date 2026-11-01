#!/usr/bin/env bash
cd /workspace/fasterwam_a3_bridge/bridge
until grep -q R2_C0_DONE /workspace/r2c0.log 2>/dev/null; do sleep 5; done
sleep 3
/opt/venv/bin/python /workspace/r2_metrics.py outputs/r2/C0 > /workspace/r2_c0_metrics.txt 2>&1
echo ALL_DONE >> /workspace/r2_c0_metrics.txt
