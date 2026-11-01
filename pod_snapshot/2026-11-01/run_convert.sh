#!/bin/bash
set -u
while [ ! -f /workspace/dagger_logs/dump_all.done ]; do sleep 3; done
source /workspace/activate_fasterwam_a3.sh
cd /workspace/fasterwam_a3_bridge/bridge || exit 1
python3 _next/dagger_to_dataset.py --takes 22:40 22:60 22:70 21:76 11:90 --out /workspace/a3_data/a3_fw_dagger > /workspace/dagger_logs/convert.log 2>&1
echo "RC=$?" >> /workspace/dagger_logs/convert.log
touch /workspace/dagger_logs/convert.done
