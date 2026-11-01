#!/bin/bash
set +e
O=/workspace/_r4_stage/pf4.txt
S=/workspace/a3_dual_arm_sim/src/a3_dual_arm_sim
echo === which python === > $O
source /workspace/activate_fasterwam_a3.sh >> $O 2>&1
which python3 >> $O 2>&1
python3 -V >> $O 2>&1
echo === camera keys === >> $O
grep -rn "observation.images" $S --include=*.py | head -12 >> $O 2>&1
grep -rni "image_height|image_width|camera_width|render_height|front" $S/config.py >> $O 2>&1
echo === yaml scene === >> $O
ls /workspace/fasterwam_a3_bridge/bridge/_next/*.yaml >> $O 2>&1
grep -rn -i "height|width|camera" /workspace/fasterwam_a3_bridge/bridge/_next/env_factory.py >> $O 2>&1
echo __PFDONE__ >> $O