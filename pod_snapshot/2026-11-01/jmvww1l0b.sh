cd /workspace/_r5_stage
grep -n "class \|import " replay_forces.py | head -30
echo SIMDEF
grep -rn "SingleCookieLowForce" /workspace/a3_dual_arm_sim/src --include=*.py | head -10