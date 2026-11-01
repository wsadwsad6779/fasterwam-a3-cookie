cd /workspace/_r5_stage
grep -n "\[7\]\|act\[7\]\|a\[7\]" replay_forces.py | head -30
echo GREP_EXPERT_OPEN_DIM
grep -n -E "gripper|GRIPPER|opening" replay_forces.py | head -40