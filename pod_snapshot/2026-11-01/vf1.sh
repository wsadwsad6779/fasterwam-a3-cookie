sed -n '/def _fail/,/def /p' /workspace/a3_dual_arm_sim/src/a3_dual_arm_sim/controllers/same_column_batch_expert.py | head -14
grep -n 'self.finished' /workspace/a3_dual_arm_sim/src/a3_dual_arm_sim/controllers/same_column_batch_expert.py | head
echo '--- cc1 ---'
date
grep -c . /workspace/cc1.out
grep -o 'seed": [0-9]*, "variant": [0-9]*, "steps": [0-9]*, "stop": "[a-z_: 0-9]*"' /workspace/cc1.out | head -12
tail -c 500 /workspace/cc1.out
echo __END__
