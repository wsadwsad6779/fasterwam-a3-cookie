cd /workspace/a3_dual_arm_sim
echo ==2X10_HEAD==
sed -n '1,60p' src/a3_dual_arm_sim/controllers/cookie_2x10_expert.py
echo ==PLAN_DEF==
grep -rnE 'batch_sizes|class .*Plan|plan =' src/a3_dual_arm_sim/controllers/*.py src/a3_dual_arm_sim/data/collect_cookie_2x10_cli.py | head -25
echo ==CLI_ARGS==
grep -nE 'add_argument|def main|Expert(' src/a3_dual_arm_sim/data/collect_cookie_2x10_cli.py | head -40
echo DONE