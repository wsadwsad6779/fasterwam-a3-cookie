#!/bin/bash
cd /workspace/fasterwam_a3_bridge/bridge
source /workspace/activate_fasterwam_a3.sh >/dev/null 2>&1
rm -f /workspace/dagger_logs/check.done
python3 _next/check_a3_dagger_dataset.py > /workspace/dagger_logs/check_ds.log 2>&1
touch /workspace/dagger_logs/check.done
