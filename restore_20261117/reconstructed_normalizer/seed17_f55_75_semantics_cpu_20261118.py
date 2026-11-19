"""Independent read-only CPU arithmetic; no model, simulator, network or controls."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
import json
import hashlib
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
ROOT = Path(__file__).resolve().parent
P = ROOT/'forensics_download'
BASE = ROOT.parent
REPO = BASE.parent
s = np.load(P/'seed17_steps.npz', allow_pickle=False)
c = np.load(P/'seed17_original_chunks.npz', allow_pickle=False)
o = np.load(P/'cpu_seed17_state_force.npz', allow_pickle=False)
u = np.load(P/'training_source_a3_fw_arrays.npz', allow_pickle=False)
cal = json.loads((BASE/'step_040000/force_calib_v1.json').read_text())
pose = o['observation.eef_pose'][:,:7].astype(float)
rot = Rotation.from_quat(pose[:,[4,5,6,3]])
R = rot.as_matrix()
force = o['observation.force'][:,cal['channels']]
opening = o['observation.state'][:,7]
dp = np.diff(pose[:,:3],axis=0)
dr = (rot[1:]*rot[:-1].inv()).as_rotvec()
valid = np.ones(len(u['proprio']),bool)
valid[u['ep_start']+u['ep_len']-1] = False
train_dp = np.zeros((len(valid),3))
train_dp[:-1] = np.diff(u['proprio'][:,:3].astype(float),axis=0)
lo,hi = pose[55:76,2].min(),pose[55:76,2].max()
matched = valid & (u['proprio'][:,2]>=lo) & (u['proprio'][:,2]<=hi)
train_rotation = Rotation.from_rotvec(u['proprio'][:,3:6].astype(float))
train_dr = np.zeros((len(valid),3))
train_dr[:-1] = (train_rotation[1:]*train_rotation[:-1].inv()).as_rotvec()

def summary(x):
    x=np.asarray(x)
    return {'n':len(x),'quantiles_0_1_50_95_99_100':np.percentile(x,[0,1,50,95,99,100],axis=0)} if len(x) else {'n':0}

def pct(x,ref):
    return np.mean(np.asarray(ref)<=x,axis=0)*100

rows=[]
for i in range(54,75):
    ci,ri=i//10,i%10
    a=c[f'chunk_{ci:03d}_pm1'][ri].astype(float)
    den=c[f'chunk_{ci:03d}_denorm'].reshape(-1,7)[ri]
    cr=c[f'chunk_{ci:03d}_rot_flat'].reshape(3,3)
    pre=np.r_[a[:3]*2, cr.T@(a[3:6]*.5)/.1]
    expected=np.clip(pre,-1,1)
    handed=s['handed_14d'][i,:6]
    local=handed[3:6]*.1
    # Nominal physical rotation: q_target=q_base*exp(local), not world pre-multiply.
    # R[i] is only a measured-base comparison; actual env base uses last-applied FK.
    world_measured_base=R[i]@local
    last=s['ik_last_applied_joint_16d'][i]
    target=s['ik_joint_target_16d'][i]
    applied=s['applied_joint_16d'][i]
    armids=np.r_[0:7,8:15]
    requested_joint=target[armids]-last[armids]
    actual_joint=applied[armids]-last[armids]
    rows.append({'i':i,'f_before':i,'f_after':i+1,'chunk':ci,'row':ri,
        'denorm7':den,'pm1_stage7':a,'adapter_preclip6':pre,'adapter6':handed,
        'adapter_reproduction_maxerr':np.max(np.abs(expected-handed)),
        'adapter_clipped_dims':np.flatnonzero(np.abs(pre)>1),
        'nominal_target_dp_world_m':handed[:3]*.025,
        'nominal_target_dr_local_rad':local,
        'dr_world_using_measured_base_rad':world_measured_base,
        'chunk_rotation_vs_measured_deg':np.degrees((Rotation.from_matrix(cr)*rot[i].inv()).magnitude()),
        'measured_dp_world_m':dp[i],'measured_dr_world_rad':dr[i],
        'rotation_target_delta_error_measured_base_rad':dr[i]-world_measured_base,
        'eef_before_xyz':pose[i,:3],'eef_after_xyz':pose[i+1,:3],
        'eef_quat_wxyz_before':pose[i,3:],'eef_quat_wxyz_after':pose[i+1,3:],
        'opening_command':s['handed_14d'][i,6], 'opening_before':opening[i], 'opening_after':opening[i+1],
        'gripper_target_applied':applied[7], 'force_before':force[i], 'force_after':force[i+1],
        'ncon_after':s['ncon'][i],
        'ik_joint_delta_max_rad':np.max(np.abs(requested_joint)),
        'applied_joint_delta_max_rad':np.max(np.abs(actual_joint)),
        'joint_rate_limit_ratio':np.linalg.norm(actual_joint)/max(np.linalg.norm(requested_joint),1e-30),
        'measured_left_qpos_rad':o['observation.state'][i+1,:7],
        'height_matched_dp_component_percentiles':pct(dp[i],train_dp[matched]),
        'height_matched_dp_norm_percentile':pct(np.linalg.norm(dp[i]),np.linalg.norm(train_dp[matched],axis=1))})

paths=[P/n for n in ['seed17_steps.npz','seed17_original_chunks.npz','cpu_seed17_state_force.npz','training_source_a3_fw_arrays.npz','forensics_identity.json']]
paths += [REPO/'bridge/action_adapter_v3.py',REPO/'finetune/prepare_a3_arrays.py', BASE/'deploy/a3_dual_arm_sim/src/a3_dual_arm_sim/sim/actions.py',BASE/'deploy/a3_dual_arm_sim/src/a3_dual_arm_sim/sim/env.py',BASE/'deploy/a3_dual_arm_sim/src/a3_dual_arm_sim/config.py',BASE/'deploy/a3_dual_arm_sim/configs/cookie_batch.yaml']
report={'NORMALIZER_IDENTITY':'RECONSTRUCTED_DETERMINISTIC_NOT_ORIGINAL','original_stats_sha256':'UNKNOWN',
 'scope':'Local CPU NumPy/SciPy arithmetic only; f55..75 means transitions i54..74. No source modifications, model/simulator/network/controls.',
 'source_sha256':{str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
 'semantics':{'training':'dp=(p[t+1]-p[t])/0.05; dr=log(R[t+1] R[t]^T)/0.5 (world), then clip first6 [-1,1]. Episode terminal duplicates excluded.',
 'bridge':'preclip dp=2*a_xyz; local_rotation=R_chunk.T*(0.5*a_rot)/0.1; elementwise clip [-1,1]. Stored chunk rotation fixed for all rows.',
 'env':'Target position = FK(last applied arm joints) + 0.025*adapter_xyz, WORLD. Target quaternion right-multiplies exp(0.1*adapter_rot), LOCAL. IK then joint trajectory rate limit; not direct measured-pose increments.',
 'units':'Unsaturated translation 2*0.025=0.05 m and rotation (0.5/0.1)*0.1=0.5 rad: intended conversions cancel, no extra scale found. Bridge clip can alter magnitude/direction. FK target base differs from measured pose.',
 'provenance_limit':'Restored local sim and sibling bridge source hashed; exact deployed bridge byte identity not recovered here. Stored numerical adapter output independently reconstructed per executed row.'},
 'availability':{'measured_arm_qpos':True,'qpos_scope':'14 arm joint positions in observation.state, plus two normalized openings; not full scene qpos or individual finger qpos.',
 'images':'Existing composed video/contact sheet present, not used to infer contact identity.',
 'camera_calibration_and_per_frame_extrinsics':'UNKNOWN','contact_pairs_and_object_clearance':'UNKNOWN','last_applied_fk_pose':'UNKNOWN'},
 'height_reference':{'z_range_m':[lo,hi],'selection':'before-step source z in exact observed f55..75 range, episode terminals excluded; no task-phase/orientation matching',
 'episodes':np.unique(u['episode_index'][matched]),'dp_world_m':summary(train_dp[matched]),
 'dp_norm_m':summary(np.linalg.norm(train_dp[matched],axis=1)),
 'measured_dr_norm_deg':summary(np.degrees(np.linalg.norm(train_dr[matched],axis=1))),
 'training_action_first6':summary(u['action'][matched,:6]),
 'training_action_saturation_count_per_dim':np.sum(np.abs(u['action'][matched,:6])>=.999999,axis=0),
 'training_dp_over_0p05_count_per_dim':np.sum(np.abs(train_dp[matched])>.05,axis=0),
 'label_vs_measured_dp_maxerr':np.max(np.abs(u['action'][matched,:3] - np.clip(train_dp[matched]/.05,-1,1)))},
 'window_summary':{'adapter_reproduction_maxerr':max(r['adapter_reproduction_maxerr'] for r in rows),
 'adapter_clip_count_per_dim':np.sum(np.abs(np.array([r['adapter_preclip6'] for r in rows]))>1,axis=0),
 'measured_dp_world_m':summary(dp[54:75]),'nominal_target_dp_world_m':summary(s['handed_14d'][54:75,:3]*.025),
 'measured_dr_deg':summary(np.degrees(np.linalg.norm(dr[54:75],axis=1)))},
 'rows':rows,
 'findings':[
 'All 21 executed first-six commands reproduce bridge conversion within 3.23e-9. Neither bridge Cartesian saturation nor arm joint rate limiting occurs in this window; source same-height saturation also zero.',
 'At i63 -> f64, opening command +1 is already sustained since i59, measured opening increases 0.9325 -> 0.9862 while nominal dz=-2.745mm and measured dz=-0.917mm. Inner pad changes 0 -> 9.219N, returns zero at f65. Opening-under-descent is observed, opening as the cause is not established.',
 'At f61..63 measured dz=-2.233,-3.034,-3.096mm with zero pads. These fit same-height source dz range [-3.479,+5.014]mm. First force excursion is not accompanied by an excessive measured descent magnitude; contact-load perturbation remains plausible.',
 'Counterevidence to a single permanent opening-collision explanation: outer pad first exceeds source at f72 with closing command -1 and rising EEF (+3.792mm); inner spike at f64 is transient.',
 'At f74..75 measured dp norm exceeds maximum of 9330 same-height source transitions, driven by lateral motion/rise, not oversized initial descent.',
 'Chunk-fixed frame is not refreshed per command: measured frame drift at i63 is 0.104 degrees, at i74 1.485 degrees. Exact FK-command-frame drift remains unknown. No evidence of a 1000x m/mm or repeated translation scale in reconstructed arithmetic.'
 ],
 'limits':['Source-height marginal comparison is not normal-contact proof; sample rows serially dependent.', 'No actual last-applied FK or contact-pair geometry: measured response cannot independently establish correct target tracking or identify contact object.', 'Quaternion frame conversion uses wxyz recorded convention; no Euler-angle subtraction.', 'No causal gripper intervention: opening and pad coincidence cannot separate opening collision from simultaneous commanded descent.'],
 'next_bounded_validation':['Offline only: recover deployed bridge/env hash and camera calibration; FK last-applied, IK-target and measured arm qpos using matching robot model without stepping; compare 21 target/residual poses.', 'If original per-substep contact logs become available, inspect only i60..74 onset timing and anonymous pair clearance; no live control or contact identity inference from image.']}
def clean(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    raise TypeError(type(x).__name__)
output=ROOT/'SEED17_F55_75_SEMANTICS_CPU_20261118.json'
output.write_text(json.dumps(report,indent=2,default=clean,allow_nan=False),encoding='utf-8')
print(json.dumps({'height_reference':report['height_reference'],'window_summary':report['window_summary']},default=clean))
for r in rows:
    print(r['f_after'],'dp mm',np.round(r['measured_dp_world_m']*1000,3),'cmd mm',np.round(r['nominal_target_dp_world_m']*1000,3),'dr deg',round(float(np.linalg.norm(r['measured_dr_world_rad'])*180/np.pi),3),'g',r['opening_command'],'open',round(float(r['opening_before']),4),round(float(r['opening_after']),4),'pad',np.round(r['force_after'][3:5],3),'rate',round(float(r['joint_rate_limit_ratio']),4),'staleR',round(float(r['chunk_rotation_vs_measured_deg']),4),'pct',r['height_matched_dp_norm_percentile'])
