import hashlib
import json
from pathlib import Path
import numpy as np

P=Path(__file__).resolve().parent/'forensics_download'
M={'NORMALIZER_IDENTITY':'RECONSTRUCTED_DETERMINISTIC_NOT_ORIGINAL','original_stats_sha256':'UNKNOWN'}
manifest=json.loads((P/'mechanismv5_download_manifest.json').read_text())
manifest+=json.loads((P/'mechanismv5_followup_manifest.json').read_text())
manifest+=json.loads((P/'mechanismv5_onset_manifest.json').read_text())
manifest+=json.loads((P/'mechanismv5_temporal_manifest.json').read_text())
manifest+=json.loads((P/'mechanismv5_memory_manifest.json').read_text())
assert all(hashlib.sha256((P/x['name']).read_bytes()).hexdigest()==x['sha256'] for x in manifest)
rows=[]
for p in sorted(P.glob('mechanismv5_*.npz')):
    z=np.load(p,allow_pickle=False); j=json.loads(p.with_suffix('.json').read_text())
    assert hashlib.sha256(p.read_bytes()).hexdigest()==j['npz_sha256']
    name=j['name'];seed=j['seed'];frame=j['frame']
    if name.startswith('memory_'):
        bp=P/'mechanismv5_onset_s17_f70_fixednoise_baseline.npz'
    elif name.startswith('negative_'):
        bp=P/f'mechanismv3_isolate_s{seed}_baseline.npz'
    elif name.startswith('dose_'):
        bp=P/'mechanismv5_robust_s17_f143_N202_baseline.npz'
    else:
        bp=P/(p.stem.rsplit('_',1)[0]+'_baseline.npz')
    b=np.load(bp); bj=json.loads(bp.with_suffix('.json').read_text())
    assert j['noise_draws']==bj['noise_draws']
    assert j['observation_sha256']==bj['observation_sha256']
    assert np.array_equal(z['normalized_proprio'],b['normalized_proprio'])
    assert np.array_equal(z['trace_before'][0],b['trace_before'][0])
    assert np.array_equal(z['force_history_mask'],b['force_history_mask'])
    original=z['force_history_original']; expected=original.copy(); mask=z['force_history_mask'].astype(bool)
    for spec in j.get('history_clip',[]):
        c=spec['channel']; expected[mask,c]=np.clip(expected[mask,c],spec['min'],spec['max'])
    assert np.array_equal(expected,z['force_history'])
    assert np.array_equal(original,b['force_history_original'] if 'force_history_original' in b else b['force_history'])
    before=z['trace_before'];after=z['trace_after'];v=z['trace_velocity'];d=z['trace_delta']
    def bf16(a):
        u=np.ascontiguousarray(a,dtype=np.float32).view(np.uint32)
        return ((u+32767+((u>>16)&1))&4294901760).view(np.float32)
    assert np.array_equal(after[:-1],before[1:])
    assert all(np.array_equal(bf16(before[i]+bf16(v[i]*d[i])),after[i]) for i in range(len(after)))
    assert np.array_equal(after[-1].reshape(32,7),z['raw32'])
    g=z['raw32'][:,6]
    rows.append({'name':name,'paired_inputs_verified':True,'euler_exact':True,
                 'action_absmax7':np.abs(z['raw32']).max(0).tolist(),'force_absmax6':np.abs(z['force_future_newton'].reshape(32,6)).max(0).tolist(),
                 'flips10':int(np.sum(np.diff(g[:10]>=0))),'flips32':int(np.sum(np.diff(g>=0))),
                 'history_changed_count6':np.sum(expected!=original,0).tolist(),
                 'history_changes': [{'history_row':int(i),'absolute_frame':frame-15+int(i),'channel':int(c),'before':float(original[i,c]),'after':float(expected[i,c])} for i,c in np.argwhere(expected!=original)],
                 'raw_max_abs_diff7_baseline':np.abs(z['raw32']-b['raw32']).max(0).tolist(),
                 'trace_action_absmax7':[np.abs(t.reshape(32,7)).max(0).tolist() for t in after],
                 'raw_exact_baseline':bool(np.array_equal(z['raw32'],b['raw32'])),
                 'future_exact_baseline':bool(np.array_equal(z['force_future_newton'],b['force_future_newton']))})
out={**M,'hash_verified_files':len(manifest),'records':rows,
     'scope':'Offline prediction mechanism only; no modified actions executed; corpus marginal clipping is not a validated safety or success repair'}
(P/'HISTORY_HOLDOUT_ANALYSIS_20261118.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps(out,indent=2))
