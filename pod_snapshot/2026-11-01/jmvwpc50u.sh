ls /workspace/a3_dual_arm_sim/src/a3_dual_arm_sim/controllers/
echo '--- batch_size ---'
grep -rn 'batch_size|num_batches|target_grip_force|stable_grip_force|MAX_INSERTION' /workspace/a3_dual_arm_sim/src/a3_dual_arm_sim/controllers/*.py | head -30
echo '--- expert.py head ---'
head -30 /workspace/a3_dual_arm_sim/src/a3_dual_arm_sim/controllers/expert.py
echo '--- registry ---'
grep -rn 'same_column|batch_expert|controller_type' /workspace/a3_dual_arm_sim/src/a3_dual_arm_sim --include=*.py | grep -v controllers/ | head -15
