import json
import hashlib
import copy
import random
import time
from pathlib import Path
import numpy as np
import diag_reconstructed_40k as D

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'gpu_rootcause_20261118'
OUT.mkdir(exist_ok=False)
E, factory, W, fw = D.setup_protocol()
import torch
from policy40k import noise_seed
from force_stream import ForceStream
from types import SimpleNamespace
policy, doc, calib, facts = D.build_policy_reconstructed(E, W, SimpleNamespace(cpu=False, gripper_mode='pm1'))
assert not policy.model.training
zero = np.asarray(calib['zero_load_ref_median'], dtype=np.float32)
assert zero.shape == (6,) and np.isfinite(zero).all()

def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x)))

def rng():
    return (random.getstate(), np.random.get_state(), torch.get_rng_state().clone(), [x.clone() for x in torch.cuda.get_rng_state_all()])

def restore(r):
    random.setstate(r[0]); np.random.set_state(r[1]); torch.set_rng_state(r[2]); torch.cuda.set_rng_state_all(r[3])

def cpu(x):
    return x.detach().float().cpu().numpy() if hasattr(x, 'detach') else np.asarray(x)

def stats(raw):
    g = raw[:10, 6]
    ids = np.flatnonzero((g[1:] >= 0) != (g[:-1] >= 0)) + 1
    return {'signs_first10': np.where(g >= 0, 1, -1).tolist(), 'flip_rows': ids.tolist(), 'flip_count': len(ids), 'flip_signed_jumps': (g[ids] - g[ids-1]).tolist(), 'motion6_absmax': np.abs(raw[:10,:6]).max(axis=0).tolist()}

write('identity.json', {**D.MARKERS, 'protocol': facts, 'zero_load_reference_raw6': zero, 'P1_neutral_residual6': [0]*6, 'P2_intervention': 'replace valid rows with calibration zero_load_ref_median; invalid padded rows and mask unchanged', 'source_sha256': D.sha256_file(Path(__file__)), 'original_stats_sha256':'UNKNOWN', 'fixed_observation_frames':[0,10], 'additional_noise_seeds':[101,202,303], 'scope':'input counterfactual inference only; actions never executed'})
results = []
for seed, folder in [(12, 'forensics_v2_out'), (17, 'forensics_full_wide_out')]:
    source = ROOT / folder / ('seed%d_observations.npz' % seed)
    z = np.load(source, allow_pickle=False)
    chunks = np.load(ROOT / folder / ('seed%d_original_chunks.npz' % seed), allow_pickle=False)
    for frame in [0,10]:
        if frame >= len(z['observation.state']):
            raise RuntimeError('missing real observation frame')
        obs = {k:z[k][frame] for k in z.files if k.startswith('observation.')}
        stream = ForceStream()
        for f in range(frame+1):
            stream.push(z['observation.force'][f])
        history, mask = stream.snapshot()
        p14 = E.build_proprio14(obs, W, calib)
        ns = noise_seed(seed, frame//10+1)
        base = policy._infer_kwargs(obs, history, mask, ns)
        assert base['num_inference_steps'] == 10 and base['num_video_frames'] == 9 and base['action_horizon'] == 32
        stem = 'seed%d_frame%d' % (seed,frame)
        np.savez_compressed(OUT / (stem+'_inputs.npz'), proprio14=p14, history=history, mask=mask, zero_load_reference=zero, normalized_proprio=cpu(base['proprio']), **obs, **D.MARKERS)
        write(stem+'_input_identity.json', {'source':str(source.relative_to(ROOT)), 'frame':frame, 'observation_sha256':{k:hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() for k,v in obs.items()}, 'history_sha256':hashlib.sha256(history.tobytes()).hexdigest(), 'mask_sha256':hashlib.sha256(mask.tobytes()).hexdigest()})
        common_rng = rng()
        torch.save(common_rng, OUT / (stem+'_rng.pt'))
        baseline = None
        variants = [('A0',False,False,ns),('A0_repeat',False,False,ns),('A1',True,False,ns),('A2',False,True,ns),('A3',True,True,ns)]
        if frame == 0:
            variants += [('N%d'%n,False,False,n) for n in [101,202,303]]
        for label,p1,p2,noise in variants:
            restore(common_rng)
            kw = copy.deepcopy(base)
            kw['seed'] = int(noise)
            if p1:
                neutral = np.array(p14, copy=True)
                neutral[8:14] = 0.0
                kw['proprio'] = W.normalize_proprio(neutral, policy.processor)
                assert np.array_equal(cpu(kw['proprio'])[...,:8], cpu(base['proprio'])[...,:8])
            if p2:
                h = history.copy()
                h[mask] = zero
                kw['force_history'] = torch.from_numpy(h).unsqueeze(0)
                assert torch.equal(kw['force_history_mask'], base['force_history_mask'])
            start = time.time()
            with torch.no_grad():
                pred = policy.model.infer_action(**kw)
            raw = cpu(pred['action']).reshape(32,7)
            future = cpu(pred['force_future_newton'])
            assert np.isfinite(raw).all() and np.isfinite(future).all()
            name = stem+'_'+label
            np.savez_compressed(OUT/(name+'.npz'), raw32=raw, force_future_newton=future, actual_normalized_proprio=cpu(kw['proprio']), force_history=cpu(kw['force_history']), force_history_mask=cpu(kw['force_history_mask']), noise_seed=np.asarray(noise), **D.MARKERS)
            rec = {**D.MARKERS, 'seed':seed, 'frame':frame, 'condition':label, 'noise_seed':int(noise), 'wall_seconds':time.time()-start, **stats(raw)}
            if baseline is None:
                baseline = raw.copy()
                oldkey = 'chunk_%03d_model_out' % (frame//10)
                rec['existing_raw_available'] = oldkey in chunks
                if oldkey in chunks:
                    old = chunks[oldkey].reshape(32,7)
                    rec['existing_raw_exact'] = bool(np.array_equal(old,raw))
                    rec['existing_raw_maxabsdiff'] = float(np.max(np.abs(old-raw)))
            rec['baseline_raw7_maxabsdiff'] = float(np.max(np.abs(raw[:,6]-baseline[:,6])))
            rec['baseline_motion6_rmsdiff'] = float(np.sqrt(np.mean((raw[:,:6]-baseline[:,:6])**2)))
            rec['baseline_xyz_rmsdiff'] = float(np.sqrt(np.mean((raw[:,:3]-baseline[:,:3])**2)))
            rec['baseline_sign_disagreements_first10'] = int(np.sum((raw[:10,6]>=0)!=(baseline[:10,6]>=0)))
            write(name+'.json',rec)
            results.append(rec)
            write('summary.json',{**D.MARKERS,'complete':False,'results':results})
            print(json.dumps(rec,default=str),flush=True)
write('summary.json',{**D.MARKERS,'complete':True,'results':results,'scope':'counterfactual output sensitivity, not task success or proof of failure cause'})
print('BOUNDED_DIAG_DONE',flush=True)
