find /workspace/a3_dual_arm_sim -name "*.py" | grep -i "env\|sim" | head -20
echo === 
grep -rn "terminated" /workspace/a3_dual_arm_sim --include=*.py | head -25