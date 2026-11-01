ls /workspace/a3_dual_arm_sim/src/a3_dual_arm_sim/ | head -20
echo FINDENV
find /workspace -name env.py -path "*a3*" -not -path "*ipynb*" 2>/dev/null | head -5