sed -n "185,232p" /workspace/a3_dual_arm_sim/src/a3_dual_arm_sim/controllers/batch_expert.py
echo ENVDECODE
grep -rn "action[6]\|action[13]\|gripper_action\|action\[7\]" /workspace/a3_dual_arm_sim/src/a3_dual_arm_sim/tasks/cookie_transfer.py | head -20