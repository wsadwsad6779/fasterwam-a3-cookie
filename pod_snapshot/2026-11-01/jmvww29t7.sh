F=/workspace/a3_dual_arm_sim/src/a3_dual_arm_sim/controllers/same_column_batch_expert.py
grep -n -A 22 "def _hold_command" $F
echo SERVO
grep -n -A 26 "def _servo" $F