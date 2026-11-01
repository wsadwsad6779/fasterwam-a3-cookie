export PYTHONPATH=/workspace/a3_dual_arm_sim/src
cd /workspace
echo '=== r5 tail ==='
tail -n 3 /workspace/a3_lora_r5/train.log
echo '=== sweep2 tail ==='
tail -n 40 /workspace/sweep2.out
echo '=== sweep2 dirs ==='
ls -la /workspace/a3_eval/ 2>/dev/null | grep pilot_single
echo '=== pids ==='
pgrep -af pilot_sweep | head -5
pgrep -af train_light | head -3
