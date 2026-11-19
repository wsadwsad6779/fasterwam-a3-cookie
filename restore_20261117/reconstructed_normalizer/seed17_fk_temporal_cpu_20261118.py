import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parent
helper = ROOT / 'seed17_fk_baseline_cpu_20261118.py'
ns = {'__file__': str(helper)}
exec(compile(helper.read_text(encoding='utf-8').split('paths=[xml_path,')[0], str(helper), 'exec'), ns)
fk, steps, obs = ns['fk'], ns['steps'], ns['obs']
calpath = ROOT.parent / 'step_040000/force_calib_v1.json'
cal = json.loads(calpath.read_text())
force = obs['observation.force'][:, cal['channels']]
errors = []
for f in range(len(obs['observation.state'])):
    for arm in range(2):
        p, r = fk(obs['observation.state'][f], arm)
        e = obs['observation.eef_pose'][f, arm*7:arm*7+7].astype(float)
        er = Rotation.from_quat(e[[4,5,6,3]])
        errors.append([np.linalg.norm(p-e[:3]), np.degrees((r*er.inv()).magnitude())])
errors = np.asarray(errors)
assert errors[:,0].max() < 1e-6 and errors[:,1].max() < 1e-4
assert np.array_equal(steps['ik_last_applied_joint_16d'][1:], steps['applied_joint_16d'][:-1])
rows = []
for i in range(len(steps['handed_14d'])):
    pm, rm = fk(obs['observation.state'][i], 0)
    pc, rc = fk(steps['ik_last_applied_joint_16d'][i], 0)
    pn, rn = fk(obs['observation.state'][i+1], 0)
    pi, ri = fk(steps['ik_joint_target_16d'][i], 0)
    dp = steps['handed_14d'][i,:3].astype(float)*.025
    offset = pc-pm
    rows.append({'i': i, 'frame_after': i+1, 'chunk': i//10, 'row': i%10,
        'base_offset_mm': (offset*1000).tolist(),
        'nominal_increment_mm': (dp*1000).tolist(),
        'target_from_measured_mm': ((offset+dp)*1000).tolist(),
        'measured_increment_mm': ((pn-pm)*1000).tolist(),
        'ik_target_residual_mm': float(np.linalg.norm(pi-(pc+dp))*1000),
        'tracking_residual_mm': float(np.linalg.norm(pn-(pc+dp))*1000),
        'force_before6': force[i].tolist(), 'force_after6': force[i+1].tolist()})
a = np.array([v['base_offset_mm'] for v in rows])
phases = []
for start, end in [(0,20),(20,54),(54,63),(63,70),(70,80),(80,139),(139,600)]:
    b = a[start:end]
    phases.append({'i_start_inclusive':start,'i_end_exclusive':end,
        'base_z_mm_min_median_max':np.percentile(b[:,2],[0,50,100]).tolist(),
        'base_norm_mm_min_median_max':np.percentile(np.linalg.norm(b,axis=1),[0,50,100]).tolist()})
paths = [helper, Path(__file__), calpath, ns['xml_path'], ROOT/'forensics_download/seed17_steps.npz', ROOT/'forensics_download/cpu_seed17_state_force.npz']
report = {'NORMALIZER_IDENTITY':'RECONSTRUCTED_DETERMINISTIC_NOT_ORIGINAL',
    'original_stats_sha256':'UNKNOWN','scope':'CPU analytical FK only, no simulator execution or modified control.',
    'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
    'validation':{'frames':len(obs['observation.state']),'arms':2,'max_position_error_m':float(errors[:,0].max()),'max_orientation_error_deg':float(errors[:,1].max()),'previous_command_alignment_all_noninitial_rows':True},
    'phase_summaries':phases, 'rows':rows,
    'interpretation_limits':['Offsets before first wrist/pad range crossing establish temporal precedence only. Normal loads and unobserved contacts are not excluded.', 'Tracking lag is a physical response, not by itself a controller defect. Measured-base counterfactual targets have not been executed.', 'Exact remote model byte identity UNKNOWN; analytical geometry is numerically validated at all recorded states.', 'No comparison to training command-based targets is available in this analysis.']}
out = ROOT/'SEED17_FK_TEMPORAL_CPU_20261118.json'
out.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
print(json.dumps({'validation':report['validation'],'phase_summaries':phases,'onset_rows':rows[57:65]},indent=2))
