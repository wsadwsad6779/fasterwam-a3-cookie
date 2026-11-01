export PYTHONPATH=/workspace/a3_dual_arm_sim/src
cd /workspace
echo '=== env.cookie_positions ==='
grep -rnE 'def cookie_positions' -A 12 /workspace/a3_dual_arm_sim/src/a3_dual_arm_sim | head -40
echo '=== cookie_truth_log ==='
cat /workspace/_r4_stage/cookie_truth_log.py 2>/dev/null || find /workspace -name cookie_truth_log.py | head
echo '=== metrics use of cookie_xyz ==='
grep -nE 'cookie_xyz|attached|argmin|dist' /workspace/_r4_stage/autonomy_metrics.py | head -40
