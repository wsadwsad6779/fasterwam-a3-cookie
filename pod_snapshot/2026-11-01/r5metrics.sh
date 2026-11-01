cd /workspace
while pgrep -f train_light_r5.py > /dev/null; do sleep 60; done
echo TRAINING_ENDED $(date)
for i in $(seq 1 90); do pgrep -f run_matrix.py > /dev/null && break; sleep 20; done
echo EVAL_SEEN $(date)
while pgrep -f run_matrix.py > /dev/null; do sleep 30; done
sleep 10
ls -la /workspace/a3_eval/r5_b0_c0_paired
tail -n 20 /workspace/a3_eval/r5_b0_c0_paired.log
export PYTHONPATH=/workspace/a3_dual_arm_sim/src
python3 /workspace/_r4_stage/autonomy_metrics.py /workspace/a3_eval/r5_b0_c0_paired | tee /workspace/a3_eval/r5_autonomy_metrics.txt
python3 /workspace/_r4_stage/autonomy_metrics.py /workspace/a3_eval/r4_b0_c0_paired | tee /workspace/a3_eval/r4_autonomy_metrics_rerun.txt
echo METRICS_CHAIN_DONE $(date)
