import hashlib
import json
from pathlib import Path
import numpy as np

P = Path(__file__).resolve().parent / 'forensics_download'
MARKERS = {'NORMALIZER_IDENTITY':'RECONSTRUCTED_DETERMINISTIC_NOT_ORIGINAL', 'original_stats_sha256':'UNKNOWN'}

def bf16(a):
    u = np.ascontiguousarray(a, dtype=np.float32).view(np.uint32)
    return ((u + np.uint32(32767) + ((u >> 16) & 1)) & np.uint32(4294901760)).view(np.float32)

manifest = json.loads((P/'mechanism_download_manifest.json').read_text())
assert all(hashlib.sha256((P/x['name']).read_bytes()).hexdigest()==x['sha256'] for x in manifest)
records=[]
for p in sorted(P.glob('mechanism_*.npz')):
    z=np.load(p,allow_pickle=False)
    if 'trace_before' not in z:
        continue
    before=z['trace_before']; after=z['trace_after']; v=z['trace_velocity']; d=z['trace_delta']
    errors=[float(np.abs(bf16(before[i]+bf16(v[i]*d[i]))-after[i]).max()) for i in range(10)]
    assert max(errors)==0 and np.array_equal(after[:-1],before[1:])
    assert np.array_equal(after[-1].reshape(32,7),z['raw32'])
    g=z['raw32'][:,6]; init=before[0,0,:,6]
    rec={'name':p.stem.removeprefix('mechanism_'),'euler_all_steps_exact':True,'trace_chain_exact':True,
         'g_min':float(g.min()),'g_max':float(g.max()),'g_absmax_first10':float(np.abs(g[:10]).max()),
         'motion6_absmax_first10':np.abs(z['raw32'][:10,:6]).max(axis=0).tolist(),
         'force_future_absmax_newton':float(np.abs(z['force_future_newton']).max()),
         'initial_final_sign_agreement32':float(np.mean((g>=0)==(init>=0))),
         'initial_final_gripper_correlation32':float(np.corrcoef(g,init)[0,1]),
         'per_step_gripper_absmax_first10':[float(np.abs(x[0,:10,6]).max()) for x in after],
         'per_step_flip_count_first10':[int(np.sum((x[0,1:10,6]>=0)!=(x[0,:9,6]>=0))) for x in after],
         'gripper32':g.tolist(), 'diff31':np.diff(g).tolist(),
         'normalized_proprio_min':float(z['normalized_proprio'].min()),'normalized_proprio_max':float(z['normalized_proprio'].max())}
    records.append(rec)
checks=[]
for a,b in [('mechanism_tracecheck_s12_f0_N101.npz','gpu20261118_seed12_frame0_N101.npz'),('mechanism_tracecheck_s17_f0_protocol.npz','gpu20261118_seed17_frame0_A0.npz')]:
    x=np.load(P/a);y=np.load(P/b)
    checks.append({'instrumented':a,'uninstrumented':b,'raw_exact':bool(np.array_equal(x['raw32'],y['raw32'])),'future_force_exact':bool(np.array_equal(x['force_future_newton'],y['force_future_newton']))})
assert all(x['raw_exact'] and x['future_force_exact'] for x in checks)
summary={**MARKERS,'hash_verified_files':len(manifest),'sampling_records':len(records),'instrumentation_checks':checks,'records':records,
         'interpretation':{'excluded':'action Euler sign/update/reordering bug from recorded scheduler input through final raw output; all 10 steps reproduced exactly in independent NumPy BF16 arithmetic',
                           'not_excluded':'upstream velocity prediction errors, data supervision problems, learned distribution errors, cross-stream noise correlations, numerical discretization error of the intended 10-step solver',
                           'state_result':'flip count does not monotonically increase with state deviation; severe states may have same-sign but very large continuous raw gripper outputs'}}
(P/'MECHANISM_TRACE_ANALYSIS_20261118.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps({'verified_files':len(manifest),'exact_euler_records':len(records),'instrumentation_checks':checks,'state_ranges':[(r['name'],r['g_min'],r['g_max']) for r in records if r['name'].endswith('protocol')]},indent=2))
