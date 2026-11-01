export PYTHONPATH=/workspace/a3_dual_arm_sim/src
cd /workspace
echo '=== settle check code ==='
sed -n 480,530p /workspace/a3_dual_arm_sim/src/a3_dual_arm_sim/controllers/same_column_batch_expert.py
echo '=== final cookie geometry ==='
python3 - <<'PY'
import numpy as np
z = np.load('/workspace/a3_eval/pilot_single_E_rows_lift0012_f045/expert_single_6001.npz')
c = z['cookie_xyz']; c0 = z['cookie_xyz0']; it = z['in_target']
fin = c[-1]
print('half', None)
for i in range(len(fin)):
    print(i, 'start', np.round(c0[i],4).tolist(), 'end', np.round(fin[i],4).tolist(), 'in_target_lastframe', int(it[-1]))
print('z unique', np.round(np.unique(np.round(fin[:,2],4)),4))
PY
echo '=== slots ==='
python3 - <<'PY'
import os
os.environ.setdefault('A3_CONFIG_PATH','/workspace/a3_dual_arm_sim/configs/cookie_batch.yaml')
from a3_dual_arm_sim.config import load_config
print('nothing')
PY
echo '=== video size sample ==='
ls -la /workspace/a3_eval/r4_b0_c0_paired/*.mp4 2>/dev/null | head -5
du -sh /workspace/a3_eval/pilot_single_* 2>/dev/null | tail -8
tail -3 /workspace/sweep2.out
