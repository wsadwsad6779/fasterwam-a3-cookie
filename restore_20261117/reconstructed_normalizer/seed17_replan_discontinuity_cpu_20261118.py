import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
P = ROOT/'forensics_download'
chunks = np.load(P/'seed17_original_chunks.npz',allow_pickle=False)
steps = np.load(P/'seed17_steps.npz',allow_pickle=False)
rows = []
for boundary in range(10,len(steps['handed_14d']),10):
    k = boundary//10
    old = chunks[f'chunk_{k-1:03d}_pm1'].astype(float)
    new = chunks[f'chunk_{k:03d}_pm1'].astype(float)
    oa = chunks[f'chunk_{k-1:03d}_adapter_out']
    na = chunks[f'chunk_{k:03d}_adapter_out']
    assert np.array_equal(oa[9],steps['handed_14d'][boundary-1])
    assert np.array_equal(na[0],steps['handed_14d'][boundary])
    rows.append({'i':boundary,'old_row9_actual_pm1':old[9].tolist(),
        'old_row10_unexecuted_pm1':old[10].tolist(),'new_row0_actual_pm1':new[0].tolist(),
        'old_row9_to_old_row10_gripper_flip':bool(np.sign(old[9,6])!=np.sign(old[10,6])),
        'old_row9_to_new_row0_actual_gripper_flip':bool(np.sign(old[9,6])!=np.sign(new[0,6])),
        'old_row10_to_new_row0_gripper_disagreement':bool(np.sign(old[10,6])!=np.sign(new[0,6])),
        'old_row10_to_new_row0_translation_mm':((new[0,:3]-old[10,:3])*.05*1000).tolist(),
        'old_row10_to_new_row0_rotation_world_rad':((new[0,3:6]-old[10,3:6])*.5).tolist()})
summary = {key:sum(v[key] for v in rows) for key in ['old_row9_to_old_row10_gripper_flip','old_row9_to_new_row0_actual_gripper_flip','old_row10_to_new_row0_gripper_disagreement']}
actual_gripper = steps['handed_14d'][:,6]
flips = np.flatnonzero(actual_gripper[1:] != actual_gripper[:-1])+1
summary.update({'actual_gripper_flip_count':len(flips),
    'actual_boundary_flip_count':int(np.sum(flips%10==0)),
    'actual_within_chunk_flip_count':int(np.sum(flips%10!=0)),
    'actual_flip_indices_before_first_wrist_pad_crossing':flips[flips<64].tolist()})
report = {'NORMALIZER_IDENTITY':'RECONSTRUCTED_DETERMINISTIC_NOT_ORIGINAL','original_stats_sha256':'UNKNOWN',
    'scope':'CPU arithmetic on stored seed17 original chunks; no inference or execution.',
    'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),P/'seed17_original_chunks.npz',P/'seed17_steps.npz']},
    'actual_boundary_rows_verified':len(rows),'summary':summary,'rows':rows,
    'limits':['Old row10 was not executed. Its difference from new row0 is a replanning-output comparison, not measured trajectory.', 'Observation, history and noise change together at recorded boundaries; this audit does not isolate their individual causal effects.', 'Translation conversion is nominal pre-Adapter saturation. Rotation differences are WORLD rotation-vector command differences, not geodesic pose differences.']}
(ROOT/'SEED17_REPLAN_DISCONTINUITY_CPU_20261118.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
print(json.dumps({'actual_boundary_rows_verified':len(rows),'summary':summary,'first_boundaries':rows[:8]},indent=2))
