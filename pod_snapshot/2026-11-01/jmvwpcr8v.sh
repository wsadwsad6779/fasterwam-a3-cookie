ls /workspace/a3_dual_arm_sim/src/a3_dual_arm_sim/workflows/ /workspace/a3_dual_arm_sim/scripts/ 2>/dev/null
echo '=== controller registry ==='
grep -rnE 'same_column|SameColumn|TransferExpert|controller' /workspace/a3_dual_arm_sim/src/a3_dual_arm_sim/workflows/*.py | head -25
echo '=== batch_size prop ==='
sed -n '100,120p' /workspace/a3_dual_arm_sim/src/a3_dual_arm_sim/controllers/cookie_2x10_expert.py
