sed -n "170,196p" /workspace/fasterwam_a3_bridge/bridge/_next/run_matrix.py
echo ===
grep -rn -E "force18|FORCE_KEY|pad_force" /workspace/a3_dual_arm_sim --include=*.py 2>/dev/null | head -20