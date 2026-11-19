"""Independent local NumPy evidence extraction. Never imports model/simulator/network."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import json
import hashlib
from pathlib import Path
import numpy as np

NORMALIZER_IDENTITY = 'RECONSTRUCTED_DETERMINISTIC_NOT_ORIGINAL'
original_stats_sha256 = 'UNKNOWN'
ROOT = Path(__file__).resolve().parent
P = ROOT / 'forensics_download'
OUT = ROOT / 'SEED17_EARLY_ONSET_CPU_20261118.json'

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def first(mask):
    a = np.flatnonzero(mask)
    return int(a[0]) if len(a) else None

def clean(a):
    if isinstance(a, np.ndarray): return a.tolist()
    if isinstance(a, np.generic): return a.item()
    raise TypeError(type(a).__name__)

s = np.load(P/'seed17_steps.npz', allow_pickle=False)
c = np.load(P/'seed17_original_chunks.npz', allow_pickle=False)
o = np.load(P/'cpu_seed17_state_force.npz', allow_pickle=False)
u = np.load(P/'training_source_a3_fw_arrays.npz', allow_pickle=False)
f = np.load(P/'training_source_a3_fw_force_arrays.npz', allow_pickle=False)
calpath = ROOT.parent/'step_040000/force_calib_v1.json'
cal = json.loads(calpath.read_text())
assert sha(P/'training_source_a3_fw_force_arrays.npz') == cal['source_sha256']['force_arrays']
ref = json.loads((ROOT/'INPUT_OOD_ANALYSIS_20261118.json').read_text())['training_reference']
channels = cal['channels']
force = o['observation.force'][:, channels]
trainforce = f['force18'][:, channels]
pose = o['observation.eef_pose'][:, :7].astype(np.float64)
q = pose[:, 3:7]; q = q/np.linalg.norm(q, axis=1, keepdims=True)
w = np.clip(q[:, 0], -1, 1); den = np.sqrt(np.maximum(0, 1-w*w))
rot = q[:, 1:]*np.divide(2*np.arccos(w), den, out=np.zeros_like(w), where=den>=1e-8)[:, None]
opening = o['observation.state'][:, 7]
p8 = np.column_stack([pose[:, :3], rot, .04*np.clip(opening, 0, 1), -.04*np.clip(opening, 0, 1)]).astype(np.float32)
assert np.array_equal(trainforce.min(0), np.array(ref['raw_force6']['min'], dtype=np.float32))
assert np.array_equal(u['proprio'].min(0), np.array(ref['proprio8']['min'], dtype=np.float32))
assert np.array_equal(s['frame'], np.arange(1, 601))
assert np.array_equal(s['executed_chunk'], np.arange(600)//10)
assert np.array_equal(s['executed_chunk_row'], np.arange(600)%10)
assert np.array_equal(s['handed_14d'], s['requested_14d'])
assert np.allclose(s['opening_after'], opening[1:], atol=1e-6, rtol=0)
raw = np.vstack([c[f'chunk_{i:03d}_model_out'][:10] for i in range(60)])
adapter = np.vstack([c[f'chunk_{i:03d}_adapter_out'][:10] for i in range(60)])
assert np.array_equal(adapter, s['handed_14d'])

def crossing(a, train, names):
    lo, hi = train.min(0), train.max(0)
    m = (a < lo-1e-6) | (a > hi+1e-6)
    result=[]
    for j, name in enumerate(names):
        k=first(m[:,j])
        result.append({'name':name,'dimension0':j,'first_frame_tolerance1e6':k,'first_frame_strict':first((a[:,j]<lo[j])|(a[:,j]>hi[j])), 'source_min':float(lo[j]),'source_max':float(hi[j]),'value_at_first':None if k is None else float(a[k,j])})
    return m, result
fm, fc = crossing(force, trainforce, ['fx','fy','fz','pad_inner','pad_outer','gripper_act'])
pm, pc = crossing(p8, u['proprio'], ['x','y','z','axis_angle_x','axis_angle_y','axis_angle_z','finger1_proxy','finger2_proxy'])
om, oc = crossing(opening[:,None], u['opening'][:,None], ['measured_normalized_opening'])
flips = np.flatnonzero(adapter[1:,6] != adapter[:-1,6])+1
angle = np.degrees(2*np.arccos(np.clip(np.abs(q @ q[0]), 0, 1)))

def executed(i):
    if not 0 <= i < 600: return None
    ci, ri = int(s['executed_chunk'][i]), int(s['executed_chunk_row'][i])
    return {'control_step0':i,'chunk':ci,'row':ri,'observation_before':i,'observation_after':i+1,
            'raw_model_out7':raw[i], 'denorm7':c[f'chunk_{ci:03d}_denorm'].reshape(-1,7)[ri],
            'pm1_stage7':c[f'chunk_{ci:03d}_pm1'][ri], 'handed14':s['handed_14d'][i],
            'requested14':s['requested_14d'][i], 'applied_joint16':s['applied_joint_16d'][i],
            'ik_joint_target16':s['ik_joint_target_16d'][i], 'opening_before':opening[i],'opening_after':opening[i+1]}

frames = {0,1,2,3,9,10,11,20,30,40,50,60,63,64,70,78,129,143}
frames.update(r['first_frame_tolerance1e6'] for r in fc+pc+oc if r['first_frame_tolerance1e6'] is not None)
frames.update([first(fm.any(1)),first(pm[:,:3].any(1)),first(pm[:,3:6].any(1))])
frames.discard(None)
events=[]
for k in sorted(frames):
    events.append({'frame':k,'time_s':o['time'][k], 'force6_measured':force[k], 'force_outside_dims0':np.flatnonzero(fm[k]),
                   'proprio8_reconstructed_from_measurement':p8[k], 'proprio_outside_dims0':np.flatnonzero(pm[k]),
                   'opening_measured':opening[k], 'rotation_deg_relative_reset':angle[k], 'contact_count':None if k==0 else s['ncon'][k-1],
                   'preceding_actually_executed':executed(k-1), 'next_actually_executed':executed(k)})
old = json.loads((P/'CPU_BOUNDARY20_SEED17_20261118.json').read_text())
legacy = json.loads((P/'failure_timeline.json').read_text())
paths = [P/n for n in ['seed17_steps.npz','seed17_original_chunks.npz','cpu_seed17_state_force.npz','seed17_meta.json','training_source_a3_fw_arrays.npz','training_source_a3_fw_force_arrays.npz','CPU_BOUNDARY20_SEED17_20261118.json','failure_timeline.json','full_artifact_analysis.json']]+[calpath, ROOT/'INPUT_OOD_ANALYSIS_20261118.json']
report={'NORMALIZER_IDENTITY':NORMALIZER_IDENTITY,'original_stats_sha256':original_stats_sha256,
        'scope':'Local CPU NumPy only; original assets opened read-only. Only this independent script and JSON written. No inference, GPU, network, noise/label studies.',
        'sources_sha256':{str(p.resolve()):sha(p) for p in paths},
        'range_contract':'42681-row calibration-hash-matched upstream source marginal bounds, not recovered original statistics. epsilon=1e-6; strict first crossings also retained. Pose uses bridge quaternion wxyz-to-axis-angle convention; fingers are proxies, not measured Panda joints.',
        'index_contract':'Observation f follows command i=f-1; actual chunk=i//10,row=i%10. All 600 rows audited; chunk row10..31 never executed.',
        'checks':{'600_handed_equals_adapter_and_requested':True,'600_frame_and_chunk_row_mappings_exact':True,'opening_matches_steps':True,'force_source_sha_matches_calib':True,'source_ranges_match_INPUT_OOD':True},
        'force_first_crossing_per_channel':fc,'proprio_first_crossing_per_dimension':pc,'opening_first_crossing':oc,
        'first_any_force_frame':first(fm.any(1)), 'first_position_frame':first(pm[:,:3].any(1)), 'first_axis_angle_frame':first(pm[:,3:6].any(1)),
        'gripper_switches_first100':[{**executed(int(i)), 'previous_adapter_gripper':adapter[i-1,6]} for i in flips if i<100],
        'first20_opening':opening[:21], 'boundary20_summary':old['summary'], 'old_chunk0_row10_unexecuted':old['boundary']['old_unexecuted_chunk0_row10'],
        'legacy_seed17_descriptive_thresholds':[r for r in legacy if r['seed']==17][0]['contact_first_frame_absN'],
        'events':events,
        'unverified':['Original normalizer unavailable.', 'Original per-call full force-history/mask and normalized proprio tensors not captured; history can only be reconstructed from observations.', 'Axis-angle marginal excursion is representation dependent; physical orientation support cannot be inferred from source marginal extrema.', 'Applied_joint16 is recorded applied joint command, not measured motion; observations are measurements.', 'No object-relative clearance, contact pair identities or sub-control-step contact onset in these archives; force excursion alone is not proof of object contact.', 'Recorded pred_force in meta is forecast, never substituted for measured force18. Source min/max is not a task-failure threshold.']}
OUT.write_text(json.dumps(report,indent=2,default=clean,allow_nan=False),encoding='utf-8')
print(json.dumps({k:report[k] for k in ['first_any_force_frame','first_position_frame','first_axis_angle_frame','force_first_crossing_per_channel','proprio_first_crossing_per_dimension','opening_first_crossing']},default=clean,indent=2))
print('EVENTS')
for e in events:
    k=e['frame']; a=e['preceding_actually_executed']; print(k,'t',round(float(e['time_s']),4),'force',np.round(e['force6_measured'],5),'p',np.round(p8[k,:6],6),'opening',round(float(opening[k]),6),'prior',None if a is None else (a['chunk'],a['row'],float(a['handed14'][6])))
print('OUTPUT',str(OUT.resolve()))
