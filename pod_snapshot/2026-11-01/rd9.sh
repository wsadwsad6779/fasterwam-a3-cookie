export PYTHONPATH=/workspace/a3_dual_arm_sim/src
cd /workspace
echo '=== config geometry ==='
grep -nE 'target|source|bin|cookie_count|slot' /workspace/a3_dual_arm_sim/configs/cookie_batch.yaml | head -40
echo '=== env props ==='
grep -nE 'def cookie_positions|cookies_in_target|TARGET_SLOTS_LOCAL|_target_bin_pos|def _in_target|upright' /workspace/a3_dual_arm_sim/src/a3_dual_arm_sim/sim/env.py | head -30
echo '=== task cfg ==='
python3 - <<'PY'
import numpy as np
from a3_dual_arm_sim.config import load_config
c = load_config('/workspace/a3_dual_arm_sim/configs/cookie_batch.yaml')
ct = c.cookie_transfer
print('src', np.round(np.asarray(ct.cookie_source_positions_m),4).tolist())
print('bin_pose', getattr(ct,'target_bin_pose_m',None), getattr(ct,'target_bin_position_m',None))
for k in dir(ct):
    if k.startswith('target') or 'slot' in k or 'bin' in k:
        print(k, getattr(ct,k))
PY
echo '=== time series cookie0 ==='
python3 - <<'PY'
import numpy as np
z = np.load('/workspace/a3_eval/pilot_single_E_rows_lift0012_f045/expert_single_6001.npz')
c = z['cookie_xyz']
for t in [0,300,600,900,1200,1500,1800,2100,2400,2759]:
    row = c[t]
    print(t, 'c0', np.round(row[0],4).tolist(), 'c1', np.round(row[1],4).tolist(), 'c8', np.round(row[8],4).tolist(), 'n_y>0.2', int((row[:,1]>0.2).sum()), 'in_target', int(z['in_target'][t]))
PY
