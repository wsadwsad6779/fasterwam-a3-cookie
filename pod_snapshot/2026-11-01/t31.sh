cd /workspace/a3_dual_arm_sim/src/a3_dual_arm_sim/controllers
sed -n '255,300p' batch_expert.py
echo ==SLOTIDX==
grep -rnE 'target_slot_index|TARGET_SLOTS_LOCAL' batch_expert.py same_column_batch_expert.py cookie_2x10_expert.py | head -15
echo ==SWEEP==
tail -c 700 /workspace/sweep.out
echo DONE