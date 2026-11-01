export PYTHONPATH=/workspace/a3_dual_arm_sim/src
cd /workspace
echo '=== sweep2 script ==='
cat /workspace/_r5_stage/pilot_sweep2.py
echo '=== select_batch ==='
sed -n 90,170p /workspace/a3_dual_arm_sim/src/a3_dual_arm_sim/controllers/same_column_batch_expert.py
