import copy
import hashlib
import json
import random
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import diag_reconstructed_40k as D

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'gpu_mechanism_v4_20261118'
OUT.mkdir(exist_ok=False)

def write(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2, default=str), encoding='utf-8')

E, factory, W, fw = D.setup_protocol()
import torch
from force_stream import ForceStream
from policy40k import noise_seed
policy, doc, calib, facts = D.build_policy_reconstructed(E, W, SimpleNamespace(cpu=False, gripper_mode='pm1'))
assert facts['device'] == 'cuda' and not policy.model.training
write('identity.json', {**D.MARKERS, 'protocol':facts, 'source_sha256':D.sha256_file(Path(__file__)), 'scope':'offline only, no counterfactual execution', 'hook':'scheduler.step read-only input/output snapshots; original return unchanged'})
archives = {}
trace = []
scheduler = policy.model.infer_action_scheduler
original_step = scheduler.step

def observed_step(*args, **kwargs):
    result = original_step(*args, **kwargs)
    if args[2].shape[-1] == 7:
        trace.append({'before':args[2].detach().float().cpu().numpy().copy(), 'velocity':args[0].detach().float().cpu().numpy().copy(), 'after':result.detach().float().cpu().numpy().copy(), 'delta':float(args[1])})
    return result

scheduler.step = observed_step
results = []

def run(job):
    seed, frame, ns = int(job['seed']), int(job['frame']), int(job['noise'])
    name = job['name']
    if (OUT / (name+'.npz')).exists():
        raise RuntimeError('refusing overwrite '+name)
    folder = 'forensics_v2_out' if seed == 12 else 'forensics_full_wide_out'
    if seed not in archives:
        archives[seed] = np.load(ROOT / folder / f'seed{seed}_observations.npz', allow_pickle=False)
    z = archives[seed]
    obs = {k:z[k][frame] for k in z.files if k.startswith('observation.')}
    stream = ForceStream()
    for f in range(max(0, frame-15), frame+1):
        stream.push(z['observation.force'][f])
    history, mask = stream.snapshot()
    history_original = history.copy()
    if 'history_clip' in job:
        for spec in job['history_clip']:
            channel = int(spec['channel'])
            history[mask.astype(bool), channel] = np.clip(history[mask.astype(bool), channel], spec['min'], spec['max'])
    kw = policy._infer_kwargs(obs, history, mask, ns)
    steps = int(job.get('steps', 10))
    kw['num_inference_steps'] = steps
    random.seed(9081); np.random.seed(9081); torch.manual_seed(9081); torch.cuda.manual_seed_all(9081)
    trace.clear()
    start = time.time()
    original_randn = torch.randn
    draws = []
    def isolated_randn(*args, **kwargs):
        shape = tuple(args[0]) if len(args) == 1 and isinstance(args[0], (tuple, list)) else tuple(args)
        stream_name = 'action' if shape == (1,32,7) else ('force' if shape == (1,32,6) else 'video')
        local = dict(kwargs)
        key = stream_name + '_noise'
        if key in job:
            local['generator'] = torch.Generator(device=local.get('device', 'cpu')).manual_seed(int(job[key]))
        result = original_randn(*args, **local)
        draws.append({'stream':stream_name, 'shape':list(result.shape), 'sha256':hashlib.sha256(result.detach().float().cpu().numpy().tobytes()).hexdigest(), 'override_seed':job.get(key)})
        return result
    torch.randn = isolated_randn
    try:
        with torch.no_grad():
            pred = policy.model.infer_action(**copy.deepcopy(kw))
    finally:
        torch.randn = original_randn
    raw = pred['action'].detach().float().cpu().numpy().reshape(32,7)
    future = pred['force_future_newton'].detach().float().cpu().numpy()
    assert len(trace)==steps and np.isfinite(raw).all() and np.isfinite(future).all()
    assert np.array_equal(trace[-1]['after'].reshape(32,7), raw)
    payload = {'raw32':raw, 'force_future_newton':future, 'force_history':history, 'force_history_original':history_original, 'force_history_mask':mask,
               'normalized_proprio':kw['proprio'].detach().float().cpu().numpy(),
               'noise_seed':np.asarray(ns), 'trace_before':np.stack([x['before'] for x in trace]),
               'trace_after':np.stack([x['after'] for x in trace]), 'trace_velocity':np.stack([x['velocity'] for x in trace]),
               'trace_delta':np.asarray([x['delta'] for x in trace]), **obs, **D.MARKERS}
    np.savez_compressed(OUT/(name+'.npz'), **payload)
    g=raw[:,6]; flips=np.flatnonzero((g[1:]>=0)!=(g[:-1]>=0))+1
    rec={**D.MARKERS, **job, 'noise_draws':draws, 'wall_seconds':time.time()-start,
         'raw_gripper32':g.tolist(), 'diff_gripper31':np.diff(g).tolist(), 'signs_first10':np.where(g[:10]>=0,1,-1).tolist(),
         'flip_rows32':flips.tolist(), 'flip_count_first10':int((flips<10).sum()), 'large_jumps_first10':int((np.abs(np.diff(g[:10]))>=0.6).sum()),
         'trace_flip_counts_first10':[int(np.sum(np.diff(x['after'].reshape(32,7)[:10,6]>=0)!=0)) for x in trace],
         'observation_sha256':{k:hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() for k,v in obs.items()},
         'npz_sha256':D.sha256_file(OUT/(name+'.npz'))}
    write(name+'.json',rec);results.append(rec)
    write('summary.json',{**D.MARKERS,'complete':False,'results':results})
    print(json.dumps(rec),flush=True)

initial=[]
for seed in [12,17]:
    ns=noise_seed(seed,1)
    for steps in [10,20,40]:
        initial.append({'seed':seed,'frame':0,'noise':ns,'steps':steps,'name':f'solver_s{seed}_f0_steps{steps}'})
for steps in [10,20,40]:
    initial.append({'seed':17,'frame':207,'noise':noise_seed(17,21),'steps':steps,'name':f'solver_s17_f207_steps{steps}'})
for frame in [143,207]:
    ns=noise_seed(17,frame//10+1)
    if frame == 143:
        initial.append({'seed':17,'frame':frame,'noise':ns,'steps':10,'name':f'history_s17_f{frame}_baseline'})
    initial.append({'seed':17,'frame':frame,'noise':ns,'steps':10,'name':f'history_s17_f{frame}_fzclip',
                    'history_clip':[{'channel':2,'min':-1.3804129362106323,'max':0.45449596643447876}]})
write('predeclared_jobs.json',initial)
for job in initial:
    run(job)
write('batch_ready.json',{'count':len(results),'state':'model retained for followup requests','idle_timeout_seconds':600})
print('BATCH_READY',flush=True)
last=time.monotonic()
while time.monotonic()-last<600:
    requests=sorted(OUT.glob('request_*.json'))
    pending=[p for p in requests if not p.stem.endswith('_done') and not (OUT/(p.stem+'_done.json')).exists()]
    if not pending:
        time.sleep(1)
        continue
    for p in pending:
        request=json.loads(p.read_text())
        if request.get('stop'):
            write(p.stem+'_done.json',{'stop':True})
            write('summary.json',{**D.MARKERS,'complete':True,'results':results})
            raise SystemExit(0)
        for job in request['jobs']:
            run(job)
        write(p.stem+'_done.json',{'jobs':len(request['jobs'])})
        last=time.monotonic()
write('summary.json',{**D.MARKERS,'complete':True,'reason':'idle timeout','results':results})
