cd /workspace/a3_dual_arm_sim
echo ==FOLLOWUP_USE==
grep -rn _is_followup_batch src/ | grep -v pyc
echo ==NUM_BATCHES_USE==
grep -rnE 'num_batches|batch_index ==' src/a3_dual_arm_sim/controllers/same_column_batch_expert.py src/a3_dual_arm_sim/controllers/batch_expert.py | head -25
echo ==BATCHSIZE_USE==
grep -rnE 'batch_size' src/a3_dual_arm_sim/controllers/batch_expert.py | head -20
echo ==BENCH_POLICY==
grep -nE 'policy|Expert|adapter' src/a3_dual_arm_sim/workflows/benchmark.py | head -25
echo DONE