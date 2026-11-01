echo '--- expert action composition ---'
ls /workspace/a3_dual_arm_sim/src/a3_dual_arm_sim/controllers/
grep -rn 'def act' -A12 /workspace/a3_dual_arm_sim/src/a3_dual_arm_sim/controllers/*same_column* | head -40
grep -rn 'np.array([' /workspace/a3_dual_arm_sim/src/a3_dual_arm_sim/controllers/*same_column* | head -20
echo '--- ACTION dim / keys in replay_forces ---'
grep -n 'action' /workspace/_r5_stage/replay_forces.py | head -12
echo __END__
