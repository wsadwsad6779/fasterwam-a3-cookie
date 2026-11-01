grep -n 'finished = True' -B3 /workspace/a3_dual_arm_sim/src/a3_dual_arm_sim/controllers/expert.py | head -12
echo '--- cc1 progress ---'
date
grep -c . /workspace/cc1.out
grep -o '"seed": [0-9]*, "variant": [0-9]*, "steps": [0-9]*' /workspace/cc1.out
echo __END__
