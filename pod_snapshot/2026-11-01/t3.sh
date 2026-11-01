cd /workspace/a3_dual_arm_sim
echo ==SAMECOL==
grep -nE 'batch_size|num_batches|MAX_INSERTION|target_grip_force|stable_grip|min_grip_opening|secure_grip|release_opening|def |= ' src/a3_dual_arm_sim/controllers/same_column_batch_expert.py | head -40
echo ==2X10EXPERT==
grep -nE '^class |batch_size|num_batches|force' src/a3_dual_arm_sim/controllers/cookie_2x10_expert.py | head -30
echo ==GRASPEXPERT==
grep -nE '^class |force|single|cookie_index|target' src/a3_dual_arm_sim/controllers/grasp_expert.py | head -30
echo ==EXAMPLES==
ls examples/ 2>/dev/null | head -30
ls src/a3_dual_arm_sim/examples/ 2>/dev/null | head -30
echo ==DATA==
ls src/a3_dual_arm_sim/data/
ls src/a3_dual_arm_sim/policies/
echo DONE