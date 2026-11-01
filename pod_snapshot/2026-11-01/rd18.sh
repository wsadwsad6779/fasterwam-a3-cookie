export PYTHONPATH=/workspace/a3_dual_arm_sim/src
cd /workspace
sed -n 486,530p /workspace/a3_dual_arm_sim/src/a3_dual_arm_sim/controllers/same_column_batch_expert.py
echo '=== helpers ==='
grep -rnE 'def cookie_is_upright|def _upright|def settled|def _settled|def cookie_contained|def _contained|def cookie_rest|_at_rest' /workspace/a3_dual_arm_sim/src/a3_dual_arm_sim/controllers /workspace/a3_dual_arm_sim/src/a3_dual_arm_sim/tasks | head -20
