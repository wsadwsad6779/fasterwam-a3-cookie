cd /workspace/a3_dual_arm_sim
echo ==VARIED==
sed -n '525,600p' src/a3_dual_arm_sim/controllers/same_column_batch_expert.py
echo ==BENCH_CLI_BODY==
sed -n '55,175p' src/a3_dual_arm_sim/data/collect_cookie_benchmark_cli.py
echo DONE