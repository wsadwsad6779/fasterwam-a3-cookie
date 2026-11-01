grep -rn FORCE /workspace/a3_dual_arm_sim/src --include=*.py | grep -v contracts.py | head -12
echo ===
sed -n "30,48p" /workspace/a3_dual_arm_sim/src/a3_dual_arm_sim/contracts.py