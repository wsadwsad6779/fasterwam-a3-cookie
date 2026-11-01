export PYTHONPATH=/workspace/a3_dual_arm_sim/src
cd /workspace
echo '=== probe loop ==='
sed -n 1,200p /workspace/_r5_stage/single_lowforce_probe.py
echo '=== displacement check ==='
python3 - <<'PY'
import numpy as np
z = np.load('/workspace/a3_eval/pilot_single_E_rows_lift0012_f045/expert_single_6001.npz')
c = z['cookie_xyz']
d = np.linalg.norm(c - c[0], axis=2).max(axis=0)
print('N cookies', c.shape)
print('max displacement per cookie (mm):', np.round(d*1000,1).tolist())
it = z['in_target']
print('in_target dtype', it.dtype, it.shape, 'unique tail', np.unique(it[-20:]))
PY
