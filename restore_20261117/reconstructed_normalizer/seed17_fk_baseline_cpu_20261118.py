"""Offline CPU FK only. Original assets read-only; never imports the environment."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation as R

ROOT = Path(__file__).resolve().parent
SIM = ROOT.parent / 'deploy/a3_dual_arm_sim'
OUT = ROOT / 'SEED17_FK_BASELINE_CPU_20261118.json'
IDENTITY = 'RECONSTRUCTED_DETERMINISTIC_NOT_ORIGINAL'
xml_path = SIM / 'models/a3_cookie_transfer.xml'
x = ET.parse(xml_path).getroot()
parents = {c:p for p in x.iter() for c in p}
u = ET.parse(SIM/'assets/a3/source/A3_mujoco.urdf').getroot()
steps = np.load(ROOT/'forensics_download/seed17_steps.npz', allow_pickle=False)
obs = np.load(ROOT/'forensics_download/cpu_seed17_state_force.npz', allow_pickle=False)
assert str(steps['NORMALIZER_IDENTITY']) == str(obs['NORMALIZER_IDENTITY']) == IDENTITY
names = [[f'{side}_{suffix}' for suffix in ['SHOULDER_P','SHOULDER_R','SHOULDER_Y','ELBOW_P','WRIST_R','WRIST_P','WRIST_Y']] for side in ['L','R']]
ids = [list(range(7)), list(range(8,15))]
def vec(node, name, default):
    return np.fromstring(node.get(name, default), sep=' ')
def transform(node):
    t = np.eye(4)
    t[:3,3] = vec(node,'pos','0 0 0')
    assert not any(k in node.attrib for k in ['axisangle','xyaxes','zaxis','euler']), node.attrib
    q = vec(node,'quat','1 0 0 0')
    t[:3,:3] = R.from_quat(q[[1,2,3,0]]).as_matrix()
    return t
chains=[]
checks=[]
for arm in range(2):
    site = x.find(f'.//site[@name="{["L","R"][arm]}_eef"]')
    chain=[]
    n=site
    while n.tag != 'worldbody':
        chain.append(n)
        n=parents[n]
    chain.reverse()
    joints=[j for b in chain for j in b.findall('joint')]
    assert [j.get('name') for j in joints] == names[arm]
    for j in joints:
        uj=u.find(f'joint[@name="{j.get("name")}"]')
        body=parents[j]
        assert body.get('name') == uj.find('child').get('link')
        assert np.allclose(vec(body,'pos','0 0 0'),vec(uj.find('origin'),'xyz','0 0 0'),rtol=0,atol=1e-10)
        assert np.allclose(vec(j,'axis','0 0 1'),vec(uj.find('axis'),'xyz','0 0 1'),rtol=0,atol=1e-10)
        assert np.allclose(vec(uj.find('origin'),'rpy','0 0 0'),0)
    assert site in list(chain[-2])  # EEF fixed on wrist, not moving fingers.
    chains.append(chain)
    checks.append({'site':site.attrib,'chain_bodies':[b.get('name') for b in chain if b.tag=='body'],'joints':names[arm]})

def fk(q16, arm):
    q=dict(zip(names[arm],np.asarray(q16)[ids[arm]]))
    t=np.eye(4)
    for node in chains[arm]:
        t=t@transform(node)
        for j in node.findall('joint'):
            assert j.get('type')=='hinge' and np.allclose(vec(j,'pos','0 0 0'),0)
            axis=vec(j,'axis','0 0 1'); axis=axis/np.linalg.norm(axis)
            a=np.eye(4); a[:3,:3]=R.from_rotvec(axis*q[j.get('name')]).as_matrix()
            t=t@a
    return t[:3,3],R.from_matrix(t[:3,:3])

def pose(p,r):
    return {'xyz_m':p.tolist(),'quat_wxyz':r.as_quat()[[3,0,1,2]].tolist()}
def residual(p,r,p0,r0):
    v=p-p0; rv=(r*r0.inv()).as_rotvec()
    return {'xyz_m':v.tolist(),'norm_m':float(np.linalg.norm(v)),'world_rotvec_rad':rv.tolist(),'angle_deg':float(np.degrees(np.linalg.norm(rv)))}
def stats(a):
    a=np.asarray(a)
    return {'min':np.min(a,axis=0).tolist(),'median':np.median(a,axis=0).tolist(),'max':np.max(a,axis=0).tolist(),'mean':np.mean(a,axis=0).tolist()}
paths=[xml_path,SIM/'assets/a3/source/A3_mujoco.urdf',SIM/'assets/a3/source/A3.urdf',SIM/'configs/cookie_batch.yaml']+[SIM/f'src/a3_dual_arm_sim/{n}' for n in ['sim/model.py','sim/urdf.py','sim/robot_builder.py','sim/env.py','sim/actions.py','contracts.py','config.py']]+[ROOT/'forensics_download'/n for n in ['seed17_steps.npz','cpu_seed17_state_force.npz','forensics_identity.json']]
report={'NORMALIZER_IDENTITY':IDENTITY,'original_stats_sha256':'UNKNOWN','scope':'CPU NumPy/SciPy analytical FK parsed from existing MJCF. No network/GPU/package installation/config changes/environment import/control/stepping. f55..75 transitions i54..74, validation includes boundary f54.','source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},'model_identity':{'robot':'A3 dual arm; source A3_mujoco.urdf','scene':x.get('model'),'base':x.find('.//body[@name="base_link"]').attrib,'urdf_joint_origin_axis_and_order_checks':'PASS','chains':checks,'eef_convention':'World site positions and site rotation; recorded quaternion wxyz, env.py:362-374. No fitted alignment or inferred tool transform. Site transform is explicit MJCF; robot_builder.py:62-95.','exact_remote_deployed_model_sha256':'UNKNOWN'},'record_index_note':str(steps['row_index_note']),'validation':{},'rows':[]}
validation=[]
for f in range(54,76):
    for arm in range(2):
        p,r=fk(obs['observation.state'][f],arm)
        e=obs['observation.eef_pose'][f,arm*7:arm*7+7].astype(float)
        re=R.from_quat(e[[4,5,6,3]])
        validation.append({'f':f,'arm':['L','R'][arm],**residual(p,r,e[:3],re)})
report['validation']={'tolerance_m':1e-6,'tolerance_deg':1e-4,'rows':validation,'max_position_error_m':max(v['norm_m'] for v in validation),'max_orientation_error_deg':max(v['angle_deg'] for v in validation)}
passed=report['validation']['max_position_error_m']<1e-6 and report['validation']['max_orientation_error_deg']<1e-4
report['validation']['status']='PASS' if passed else 'BLOCKED'
if passed:
    assert np.array_equal(steps['ik_last_applied_joint_16d'][54:75],steps['applied_joint_16d'][53:74])
    report['previous_command_alignment']='PASS: last[i] exactly equals applied[i-1] for all i54..74 and all16 columns'
    for i in range(54,75):
        for arm in range(2):
            pm,rm=fk(obs['observation.state'][i],arm)
            pc,rc=fk(steps['ik_last_applied_joint_16d'][i],arm)
            pa,ra=fk(steps['applied_joint_16d'][i],arm)
            pn,rn=fk(obs['observation.state'][i+1],arm)
            pik,rik=fk(steps['ik_joint_target_16d'][i],arm)
            action=steps['handed_14d'][i,arm*7:arm*7+7].astype(float)
            dp=action[:3]*.025
            local=action[3:6]*.1 if np.any(np.abs(action[3:6])>1e-4) else np.zeros(3)
            inc=R.from_rotvec(local)
            pt,rt=pc+dp,rc*inc
            ph,rh=pm+dp,rm*inc
            report['rows'].append({'i':i,'f_after':i+1,'arm':['L','R'][arm], 'measured_base':pose(pm,rm),'previous_command_base':pose(pc,rc),'command_minus_measured_base':residual(pc,rc,pm,rm),'actual_cartesian_target':pose(pt,rt),'hypothetical_measured_base_target':pose(ph,rh),'actual_minus_measured_base_target':residual(pt,rt,ph,rh),'nominal_increment_world_m':dp.tolist(),'command_FK_increment':residual(pa,ra,pc,rc),'measured_increment':residual(pn,rn,pm,rm),'command_increment_minus_measured_increment_m':((pa-pc)-(pn-pm)).tolist(),'IK_FK_minus_cartesian_target':residual(pik,rik,pt,rt),'after_measured_minus_cartesian_target':residual(pn,rn,pt,rt)})
    summary={}
    for side in ['L','R']:
        rr=[v for v in report['rows'] if v['arm']==side]
        summary[side]={'base_position_norm_m':stats([v['command_minus_measured_base']['norm_m'] for v in rr]),'base_orientation_deg':stats([v['command_minus_measured_base']['angle_deg'] for v in rr]),'base_xyz_m':stats([v['command_minus_measured_base']['xyz_m'] for v in rr]),'sum_command_increment_minus_measured_increment_m':np.sum([v['command_increment_minus_measured_increment_m'] for v in rr],axis=0).tolist(),'base_xyz_sign_counts_positive':np.sum(np.array([v['command_minus_measured_base']['xyz_m'] for v in rr])>0,axis=0).tolist()}
    report['summary']=summary
    report['i63']=[v for v in report['rows'] if v['i']==63]
    report['interpretation']={'identity_limit':'Local A3 model geometry and tool frame numerically validated against both recorded arms, but exact remote model bytes UNKNOWN.','mechanism':'Target offset is FK(previous command)-FK(measured qpos); common WORLD translation cancels in comparing targets. Command FK increment is not necessarily nominal increment (IK residual). Orientation right-multiplies local exponential, with source has_rot threshold applied.','persistence':'Use signed base residuals and endpoint sum of command-minus-measured increments, not sum of base residuals: repeated offsets are not independent accumulated drift. Observational quantification is not a causal counterfactual rollout.','training':'Previously verified labels encode measured EEF increments, not command FK increments. Here compare recorded deployment increments only; training command-increment distribution UNKNOWN.','causal_attribution_to_baseline_mismatch':'UNKNOWN','contact_pair_identity':'UNKNOWN','original_normalizer':'UNKNOWN'}
else:
    report['blocked_reason']='Measured qpos FK does not reproduce recorded EEF pose within predeclared float32 tolerances; no command-baseline/target residuals computed. No alignment or tool transform fitted.'
if passed:
    report['findings']=[
        'Left baseline offset norm median 2.166685 mm, maximum 10.258425 mm; orientation median 0.068879 deg, maximum 0.703610 deg. Right baseline residual is zero at recorded precision.',
        'At i63 actual target is 0.126453,-0.086635,-2.758875 mm relative to the measured-base target; norm 2.763130 mm. Actual target z=0.776401169 m versus measured-base target z=0.779160043 m. Nominal dz=-2.745163 mm becomes target-minus-measured z=-5.504037 mm.',
        'At i63 IK FK positional residual is 0.028087 mm, versus next measured-to-target norm 4.679216 mm. These are distinct errors; neither is an identification of contact cause.',
        'Left signed baseline dz stays negative for i57..70 (14 consecutive commands), minimum -4.569974 mm at i64; reverses positive for i71..74, reaching +10.001208 mm at i74. Thus sustained phase-dependent bias is observed, not a permanent one-direction bias.',
        'Summed command-FK increment minus measured increment over i54..74 is [1.366414,-1.521009,7.350208] mm: endpoint baseline-residual change, not evidence of runaway accumulation. Training measured-increment semantics and command-increment semantics are numerically non-equivalent on this deployment trace; causality remains UNKNOWN.'
    ]
OUT.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
print(json.dumps({'validation':{k:v for k,v in report['validation'].items() if k!='rows'},'summary':report.get('summary'),'i63':report.get('i63'),'blocked_reason':report.get('blocked_reason')},indent=2))
