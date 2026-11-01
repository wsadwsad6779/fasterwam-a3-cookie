#!/bin/bash
set +e
O=/workspace/_r4_stage/pf3.txt
S=/workspace/a3_dual_arm_sim/src/a3_dual_arm_sim
echo === grep in_target === > $O
grep -rn cookies_in_target $S --include=*.py >> $O 2>&1
echo === grep images.front === >> $O
grep -rn images.front $S --include=*.py | head -20 >> $O 2>&1
echo === grep n_cookies === >> $O
grep -rni num_cookies $S --include=*.py | head -20 >> $O 2>&1
echo === tree === >> $O
ls $S/sim $S/tasks >> $O 2>&1
echo __PFDONE__ >> $O