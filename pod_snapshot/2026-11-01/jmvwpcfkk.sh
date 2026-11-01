grep -rnE 'batch_size|num_batches|target_grip_force|stable_grip_force|MAX_INSERTION_FORCE_N|min_grip_opening' /workspace/a3_dual_arm_sim/src/a3_dual_arm_sim/controllers/ | head -25
echo '=== cookie_2x10 head ==='
head -18 /workspace/a3_dual_arm_sim/src/a3_dual_arm_sim/controllers/cookie_2x10_expert.py
echo '=== cookie_expert head ==='
head -22 /workspace/a3_dual_arm_sim/src/a3_dual_arm_sim/controllers/cookie_expert.py
echo '=== grasp_expert head ==='
head -14 /workspace/a3_dual_arm_sim/src/a3_dual_arm_sim/controllers/grasp_expert.py
