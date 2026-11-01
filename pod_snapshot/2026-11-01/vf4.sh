cd /workspace/a3_dual_arm_sim/src/a3_dual_arm_sim/controllers
grep -rn 'def _fail' . | head
grep -rn 'finished' . | head -20
echo __END__
