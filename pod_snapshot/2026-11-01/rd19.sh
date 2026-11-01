export PYTHONPATH=/workspace/a3_dual_arm_sim/src
cd /workspace
grep -rnE 'def privileged_cookie_in_target' -A 22 /workspace/a3_dual_arm_sim/src/a3_dual_arm_sim/tasks/cookie_transfer.py
grep -rnE 'def cookie_in_target|def _in_target|def is_in_target' -A 20 /workspace/a3_dual_arm_sim/src/a3_dual_arm_sim/tasks/cookie_transfer.py /workspace/a3_dual_arm_sim/src/a3_dual_arm_sim/sim/*.py | head -60
