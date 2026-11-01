grep -rn force18 /workspace/_r5_stage /workspace/fasterwam_a3_bridge /workspace/a3_dual_arm_sim/src --include=*.py 2>/dev/null | head -12
echo ===
grep -n -E "observation.force|FORCE|force" /workspace/fasterwam_a3_bridge/bridge/_next/dagger_to_dataset.py | head -12