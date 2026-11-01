ls /workspace/a3_dual_arm_sim/examples/ | head -30
grep -n add_argument /workspace/a3_dual_arm_sim/examples/collect_cookie_benchmark.py | head -30
echo '===== policy wiring ====='
grep -nE 'policy|expert|batch' /workspace/a3_dual_arm_sim/examples/collect_cookie_benchmark.py | head -25
