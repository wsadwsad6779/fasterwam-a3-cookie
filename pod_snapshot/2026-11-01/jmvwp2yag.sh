grep -rn --include=*.py 'observation.force' /workspace/a3_dual_arm_sim/src /workspace/fasterwam-a3-cookie 2>/dev/null | head -20
echo '---'
grep -rn --include=*.py -e 'FORCE_KEYS' -e 'force_names' -e 'def.*force' /workspace/a3_dual_arm_sim/src | head -30
