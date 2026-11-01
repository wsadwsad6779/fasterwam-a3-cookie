cd /workspace/_r5_stage
grep -n 'self.finished' /workspace/a3_dual_arm_sim/src/a3_dual_arm_sim/same_column_batch_expert.py | head -20
grep -n 'def _fail' -A6 /workspace/a3_dual_arm_sim/src/a3_dual_arm_sim/same_column_batch_expert.py | head -20
echo '--- run_episode loop order in replay_forces ---'
sed -n '176,215p' replay_forces.py
echo __END__
