export PYTHONPATH=/workspace/a3_dual_arm_sim/src
cd /workspace
echo '=== r4 steps/reason ==='
python3 - <<'PY'
import json
for p in ['/workspace/a3_eval/r4_b0_c0_paired/summary_lora.json']:
    for r in json.load(open(p)):
        print(r['control'], r['seed'], 'steps', r['steps'], 'replans', r['replans'], 'hard', r['estop_steps'], 'binfloor', r['binfloor_steps'], 'dur', r['duration_s'])
PY
echo '=== upright/settle check ==='
grep -rnE 'not upright' /workspace/a3_dual_arm_sim/src/a3_dual_arm_sim | head
echo '=== expert phase attr ==='
grep -nE 'self.phase|self._phase|PHASES|def act' /workspace/a3_dual_arm_sim/src/a3_dual_arm_sim/controllers/same_column_batch_expert.py | head -20
echo '=== E npz keys ==='
python3 - <<'PY'
import numpy as np
z = np.load('/workspace/a3_eval/pilot_single_E_rows_lift0012_f045/expert_single_6001.npz')
print(list(z.keys()), z['pad_force_inner'].shape)
f = np.maximum(z['pad_force_inner'], z['pad_force_outer'])
idx = np.argsort(-f)[:12][::-1]
op = z['opening']; ee = z['eef_xyz']; it = z['in_target']
for i in idx:
    print('step', i, 'padF %.2f' % f[i], 'open %.3f' % op[i], 'eef', np.round(ee[i],3).tolist(), 'in_target', int(it[i]))
print('pad p50 %.3f p90 %.3f max %.3f' % (np.percentile(f,50), np.percentile(f,90), f.max()))
print('frames over 2N', int((f>2).sum()), 'over 5N', int((f>5).sum()))
PY
