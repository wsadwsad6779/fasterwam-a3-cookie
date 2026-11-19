import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parent
helper = ROOT/'seed17_fk_baseline_cpu_20261118.py'
ns = {'__file__': str(helper)}
exec(compile(helper.read_text(encoding='utf-8').split('paths=[xml_path,')[0], str(helper), 'exec'), ns)
fk, steps, obs = ns['fk'], ns['steps'], ns['obs']
P = ROOT/'forensics_download'
cal = json.loads((ROOT.parent/'step_040000/force_calib_v1.json').read_text())
force = obs['observation.force'][:, cal['channels']].astype(float)
opening = obs['observation.state'][:,7].astype(float)
pose = obs['observation.eef_pose'][:,:3].astype(float)

rows = []
for i in range(100, 145):
    pm, rm = fk(obs['observation.state'][i], 0)
    pc, rc = fk(steps['ik_last_applied_joint_16d'][i], 0)
    pn, rn = fk(obs['observation.state'][i+1], 0)
    a = steps['handed_14d'][i].astype(float)
    rows.append({'i': i, 'chunk': i//10, 'row': i%10,
        'command6': a[:6].tolist(), 'gripper_cmd': float(a[6]),
        'nominal_dz_mm': float(a[2]*25.0),
        'base_offset_mm': ((pc-pm)*1000).tolist(),
        'target_minus_measured_dz_mm': float((pc-pm+a[:3]*.025)[2]*1000),
        'measured_dz_mm': float((pn-pm)[2]*1000),
        'z_before': float(pose[i,2]), 'z_after': float(pose[i+1,2]),
        'opening_before': float(opening[i]), 'opening_after': float(opening[i+1]),
        'pad_inner_after_N': float(force[i+1,3]), 'pad_outer_after_N': float(force[i+1,4])})

paths = [Path(__file__), P/'cpu_seed17_state_force.npz', P/'seed17_steps.npz', ROOT.parent/'step_040000/force_calib_v1.json', ns['xml_path'], ns['xml_path'].parent.parent/'assets/a3/source/A3_mujoco.urdf']
report = {'NORMALIZER_IDENTITY': 'RECONSTRUCTED_DETERMINISTIC_NOT_ORIGINAL', 'original_stats_sha256': 'UNKNOWN',
 'scope': 'CPU FK and stored-episode arithmetic around the start of the sustained jam (i100..144). No simulator, model or network.',
 'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
 'context': 'Sustained left pad contact runs f134..311 with peak338.867N; z minimum of episode0.73286m at the same interval.',
 'rows': rows,
 'limits': ['Target geometry is analytical; contact with table, box or cookie is not identified.', 'No counterfactual execution; a downward target is not by itself a collision.', 'Quaternion not needed for z comparisons here.']}
(ROOT/'SEED17_JAM_ONSET_CONTEXT_CPU_20261118.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
for r in rows:
    print(r['i'], 'z', round(r['z_after'],4), 'nom dz', round(r['nominal_dz_mm'],2), 'base dz', round(r['base_offset_mm'][2],2),
          'tgt-m dz', round(r['target_minus_measured_dz_mm'],2), 'meas dz', round(r['measured_dz_mm'],2),
          'op', round(r['opening_after'],3), 'pad', round(r['pad_inner_after_N'],2), round(r['pad_outer_after_N'],2), 'g', r['gripper_cmd'])
