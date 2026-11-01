cd /workspace/a3_dual_arm_sim/src/a3_dual_arm_sim
grep -rn 'cartesian_delta' contracts.py sim/*.py tasks/*.py config.py 2>/dev/null | head -12
echo '--- action space shape ---'
grep -rn 'action_space|ACTION_DIM|action_dim' contracts.py sim/*.py tasks/*.py 2>/dev/null | head -12
grep -rn 'def step' -A18 sim/env.py 2>/dev/null | head -40
ls sim tasks
echo __END__
