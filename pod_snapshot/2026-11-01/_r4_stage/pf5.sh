#!/bin/bash
set +e
O=/workspace/_r4_stage/pf5.txt
S=/workspace/a3_dual_arm_sim/src/a3_dual_arm_sim
echo === dims === > $O
grep -rni "image_size|images_shape|_size|height|width" $S/sim --include=*.py | head -30 >> $O 2>&1
echo === mjcf === >> $O
grep -rni "height|width|fov" $S/data 2>/dev/null | head -20 >> $O 2>&1
ls $S/sim $S/data >> $O 2>&1
echo === env obs space === >> $O
grep -rn "FRONT_IMAGE|front" $S/sim/*.py | head -30 >> $O 2>&1
echo __PFDONE__ >> $O