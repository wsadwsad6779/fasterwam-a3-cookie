F=/workspace/a3_dual_arm_sim/src/a3_dual_arm_sim/controllers/same_column_batch_expert.py
grep -n -e _insertion_jammed -e _base_push_attempts -e _rim_contact_stop -e _wait_for_approach $F
B=/workspace/a3_dual_arm_sim/src/a3_dual_arm_sim/controllers/batch_expert.py
grep -n "def reset" -A 22 $B | head -60
