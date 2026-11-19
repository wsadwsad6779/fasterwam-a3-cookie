import hashlib
import json
from pathlib import Path
import numpy as np

P = Path(__file__).resolve().parent / 'forensics_download'
M = {'NORMALIZER_IDENTITY': 'RECONSTRUCTED_DETERMINISTIC_NOT_ORIGINAL', 'original_stats_sha256': 'UNKNOWN'}
manifest = json.loads((P/'mechanismv4_download_manifest.json').read_text())
assert all(hashlib.sha256((P/x['name']).read_bytes()).hexdigest()==x['sha256'] for x in manifest)

def bf16(a):
    u = np.ascontiguousarray(a, dtype=np.float32).view(np.uint32)
    return ((u+np.uint32(32767)+((u>>16)&1))&np.uint32(4294901760)).view(np.float32)

rows=[]
for p in sorted(P.glob('mechanismv4_*.npz')):
    z=np.load(p,allow_pickle=False)
    j=json.loads(p.with_suffix('.json').read_text())
    assert hashlib.sha256(p.read_bytes()).hexdigest()==j['npz_sha256']
    n=j['steps']
    assert len(z['trace_after'])==n
    assert np.array_equal(z['trace_after'][:-1],z['trace_before'][1:])
    for i in range(n):
        assert np.array_equal(bf16(z['trace_before'][i]+bf16(z['trace_velocity'][i]*z['trace_delta'][i])),z['trace_after'][i])
    assert np.array_equal(z['trace_after'][-1].reshape(32,7),z['raw32'])
    seed,frame=j['seed'],j['frame']
    if j['name'].startswith('holdout_'):
        baseline=P/f'mechanismv4_holdout_s{seed}_f{frame}_baseline.npz'
    else:
        baseline=P/(f'mechanismv4_solver_s{seed}_f{frame}_steps10.npz' if frame in [0,207] else f'mechanismv4_history_s{seed}_f{frame}_baseline.npz')
    b=np.load(baseline)
    bj=json.loads(baseline.with_suffix('.json').read_text())
    assert j['noise_draws']==bj['noise_draws']
    assert np.array_equal(z['trace_before'][0],b['trace_before'][0])
    assert np.array_equal(z['normalized_proprio'],b['normalized_proprio'])
    assert j['observation_sha256']==bj['observation_sha256']
    assert np.array_equal(z['force_history_mask'],b['force_history_mask'])
    orig=z['force_history_original']
    assert np.array_equal(orig,b['force_history_original'])
    expected=orig.copy()
    for s in j.get('history_clip',[]):
        c=s['channel']; mask=z['force_history_mask'].astype(bool)
        expected[mask,c]=np.clip(expected[mask,c],s['min'],s['max'])
    assert np.array_equal(expected,z['force_history'])
    g=z['raw32'][:,6]
    rows.append({'name':j['name'],'steps':n,'seed':seed,'frame':frame,
                 'exact_euler':True,'paired_inputs_verified':True,'flips10':int(np.sum(np.diff(g[:10]>=0))),
                 'flips32':int(np.sum(np.diff(g>=0))), 'gripper_min':float(g.min()),'gripper_max':float(g.max()),
                 'sign_hamming32_baseline':int(np.sum((g>=0)!=(b['raw32'][:,6]>=0))),
                 'action_absmax7':np.abs(z['raw32']).max(0).tolist(),
                 'future_force_absmax6':np.abs(z['force_future_newton'].reshape(32,6)).max(0).tolist(),
                 'trace_gripper_absmax':[float(np.abs(a[...,6]).max()) for a in z['trace_after']],
                 'history_changed_count6':np.sum(z['force_history']!=orig,axis=0).tolist(),
                 'changed_channels':[s['channel'] for s in j.get('history_clip',[])]})
checks=[]
for seed,frame,old in [(12,0,'mechanismv3_isolate_s12_baseline'),(17,0,'mechanismv3_isolate_s17_baseline'),(17,207,'mechanism_s17_f207_protocol'),(17,143,'mechanism_s17_f143_protocol')]:
    new=next(x for x in rows if x['seed']==seed and x['frame']==frame and x['steps']==10 and not x['changed_channels'])
    a=np.load(P/('mechanismv4_'+new['name']+'.npz')); b=np.load(P/(old+'.npz'))
    checks.append({'new':new['name'],'old':old,'raw_exact':bool(np.array_equal(a['raw32'],b['raw32'])),'future_force_exact':bool(np.array_equal(a['force_future_newton'],b['force_future_newton']))})
assert all(c['raw_exact'] and c['future_force_exact'] for c in checks)
out={**M,'hash_verified_files':len(manifest),'baseline_exact_checks':checks,'records':rows,
     'interpretation':{'solver':'10 to 40 steps does not eliminate initial fragmentation or severe-state continuous amplification; not proof of convergence or absence of all numerical effects',
                       'history':'Clipping intervention is an offline input-path causal comparison, not executed control or a validated deployment repair; source marginal ranges do not establish joint in-distribution inputs'}}
(P/'SOLVER_HISTORY_ANALYSIS_20261118.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps({'records':len(rows),'baseline_checks':checks,'results':[{k:r[k] for k in ['name','flips10','sign_hamming32_baseline','gripper_max','action_absmax7','history_changed_count6']} for r in rows]},indent=2))
