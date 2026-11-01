cd /workspace/a3_dual_arm_sim
echo ==VARIANT_BASE==
grep -nE 'class |batch_size|num_batches|first_grasp|plan|required_cookies' src/a3_dual_arm_sim/controllers/same_column_batch_expert.py | head -30
echo ==PLAN_FILE==
sed -n '1,70p' src/a3_dual_arm_sim/tasks/cookie_2x10_plan.py
echo ==CLI2X10_ARGS==
grep -nE add_argument src/a3_dual_arm_sim/data/collect_cookie_2x10_cli.py
echo ==BENCH_CLI_EXPERT==
grep -nE 'add_argument|Expert|required_cookies|Recorder' src/a3_dual_arm_sim/data/collect_cookie_benchmark_cli.py | head -35
echo DONE