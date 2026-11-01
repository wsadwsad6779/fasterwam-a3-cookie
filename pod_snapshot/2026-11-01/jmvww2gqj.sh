D=/workspace/a3_dual_arm_sim/src/a3_dual_arm_sim/controllers
ls $D
grep -rn "def _servo\|def _hold_command\|def act" $D | head -20