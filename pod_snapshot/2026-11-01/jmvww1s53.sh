F=/workspace/a3_dual_arm_sim/src/a3_dual_arm_sim/controllers/same_column_batch_expert.py
wc -l $F
grep -n -E "action|grip|open" $F | head -40