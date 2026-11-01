#!/usr/bin/env bash
C=$1
cd /workspace/fasterwam_a3_bridge/bridge
until grep -q R2_${C}_DONE /workspace/r2${C}.log 2>/dev/null; do sleep 5; done
sleep 3
/opt/venv/bin/python /workspace/r2_metrics.py outputs/r2/$C > /workspace/r2_${C}_metrics.txt 2>&1
echo ALL_DONE >> /workspace/r2_${C}_metrics.txt
