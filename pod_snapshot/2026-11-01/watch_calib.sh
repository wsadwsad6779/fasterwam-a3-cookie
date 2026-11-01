#!/usr/bin/env bash
while pgrep -f "pip3 install" >/dev/null; do sleep 20; done
sleep 5
source /workspace/activate_fasterwam_a3.sh
cd /workspace/fasterwam_a3_bridge/bridge
python3 _next/calibrate.py --config /workspace/a3_dual_arm_sim/configs/cookie_batch.yaml > /workspace/calibrate.log 2>&1
echo CALIB_RC_$? >> /workspace/calibrate.log
