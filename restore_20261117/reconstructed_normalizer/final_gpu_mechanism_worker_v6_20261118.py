"""v6 GPU mechanism worker: real-replan counterfactuals under the reconstructed normalizer.

Differences from v5:
  * states are REAL replan boundaries f = 10*k (chunk k was actually computed from that
    observation during the recorded episode), so A0 can be compared against the recorded
    chunk_<k>_model_out;
  * P1 neutrality is an explicit intervention (proprio force dims 8..13 -> 0, i.e. z=0
    after the reconstructed normaliser's own clip), not only P2 history clipping;
  * within-P2 localisation: clip all valid history rows / only the newest / all but newest;
  * read-only forward hooks on every module whose name contains "force" record output
    max-abs, so the amplification can be located to a stage without editing the model.

Nothing here executes control, loads the simulator or writes any archived asset.
All outputs carry NORMALIZER_IDENTITY=RECONSTRUCTED_DETERMINISTIC_NOT_ORIGINAL.
"""
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
OUT = ROOT / 'gpu_mechanism_v6_20261118'
OUT.mkdir(exist_ok=False)

OBS_NPZ = ROOT / 'forensics_full_wide_out' / 'seed17_observations.npz'
CHUNK_NPZ = ROOT / 'forensics_full_wide_out' / 'seed17_original_chunks.npz'
SEED = 17
STEPS_EXPECTED = 10


def write(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2, default=str), encoding='utf-8')


E, factory, W, fw = D.setup_protocol()
import torch
from force_stream import ForceStream, CHANNELS, CHANNEL_NAMES
from policy40k import noise_seed

policy, doc, calib, facts = D.build_policy_reconstructed(
    E, W, SimpleNamespace(cpu=False, gripper_mode='pm1'))
assert facts['device'] == 'cuda' and not policy.model.training

# Source-support clip bounds for P2, in raw newtons. The recovered calibration stores the
# corpus per-channel min/max in the same channel order ForceStream uses; the saturation
# audit showed these are exactly the training-support edges.
CLIP_LO = np.asarray(calib['per_channel_min'], dtype=np.float64)
CLIP_HI = np.asarray(calib['per_channel_max'], dtype=np.float64)
assert len(CLIP_LO) == 6 and np.all(CLIP_LO < CLIP_HI)
assert list(calib['channels']) == list(CHANNELS)

OBS = np.load(OBS_NPZ, allow_pickle=False)
CHUNKS = np.load(CHUNK_NPZ, allow_pickle=False)

write('identity.json', {
    **D.MARKERS,
    'protocol': facts,
    'scope': 'offline only, no counterfactual control execution',
    'source_sha256': D.sha256_file(Path(__file__)),
    'states': 'seed17 recorded replan boundaries f=10k; observation, images, text, mask and all '
              'initial noise draws are held at the protocol values',
    'interventions': {
        'A0': 'no change',
        'P2clip_all': 'clip every mask-true history row to the corpus per-channel raw range',
        'P2clip_current': 'clip only the newest mask-true history row',
        'P2clip_history': 'clip every mask-true history row except the newest',
        'P1neutral': 'set proprio force dims 8..13 to 0 in the normalised 14-wide vector',
        'P1P2': 'P2clip_all plus P1neutral',
    },
    'clip_bounds_newton': {'lo': CLIP_LO.tolist(), 'hi': CLIP_HI.tolist(),
                           'channel_names': list(CHANNEL_NAMES)},
    'hook': 'forward hooks on modules whose name contains "force"; output max-abs only, '
            'return values untouched',
})

hook_records = {}
hook_handles = []


def _tensor_maxabs(x):
    if torch.is_tensor(x) and x.numel() > 0:
        return float(x.detach().float().abs().max().cpu())
    return None


def install_hooks(model):
    for name, mod in model.named_modules():
        if 'force' not in name.lower():
            continue

        def hook(_m, _inp, out, name=name):
            vals = []
            items = out if isinstance(out, (tuple, list)) else (out,)
            for item in items:
                v = _tensor_maxabs(item)
                if v is not None:
                    vals.append(v)
            if vals:
                rec = hook_records.setdefault(name, {'max_abs': 0.0, 'calls': 0})
                rec['max_abs'] = max(rec['max_abs'], max(vals))
                rec['calls'] += 1

        hook_handles.append(mod.register_forward_hook(hook))


install_hooks(policy.model)

scheduler = policy.model.infer_action_scheduler
original_step = scheduler.step
trace = []


def observed_step(*args, **kwargs):
    result = original_step(*args, **kwargs)
    if args[2].shape[-1] == 7:
        trace.append({'before': args[2].detach().float().cpu().numpy().copy(),
                      'velocity': args[0].detach().float().cpu().numpy().copy(),
                      'after': result.detach().float().cpu().numpy().copy(),
                      'delta': float(args[1])})
    return result


scheduler.step = observed_step

results = []
a0_cache = {}


def summarize_raw(raw32, recorded):
    first10 = raw32[:10]
    rec10 = recorded[:10]
    trans = lambda a: np.sign(np.where(np.abs(a) >= 1e-6, a, 1.0))
    return {
        'raw_absmax_32': float(np.abs(raw32).max()),
        'raw_absmax_first10': float(np.abs(first10).max()),
        'vs_recorded_full32': {
            'max_abs_diff': float(np.abs(raw32 - recorded).max()),
            'mean_abs_diff': float(np.abs(raw32 - recorded).mean()),
            'per_dim_max_abs_diff': np.abs(raw32 - recorded).max(0).tolist(),
            'bit_equal': bool(np.array_equal(raw32, recorded)),
        },
        'vs_recorded_first10': {
            'max_abs_diff': float(np.abs(first10 - rec10).max()),
            'dir_sign_agree': [int((trans(first10[:, j]) == trans(rec10[:, j])).sum()) for j in range(6)],
            'gripper_sign_agree': int((trans(first10[:, 6]) == trans(rec10[:, 6])).sum()),
            'gripper_sign': np.sign(first10[:, 6]).tolist(),
            'recorded_gripper_sign': np.sign(rec10[:, 6]).tolist(),
        },
    }


def run(job):
    name = job['name']
    if (OUT / (name + '.npz')).exists():
        raise RuntimeError('refusing overwrite ' + name)
    frame = int(job['frame'])
    chunk = frame // 10
    ns = int(job['noise'])
    assert ns == noise_seed(SEED, chunk + 1), 'noise must be the protocol replan seed'
    obs = {k: OBS[k][frame] for k in OBS.files if k.startswith('observation.')}
    stream = ForceStream()
    for f in range(max(0, frame - 15), frame + 1):
        stream.push(OBS['observation.force'][f])
    history, mask = stream.snapshot()
    history_original = history.copy()
    valid = mask.astype(bool)
    hist = history.astype(np.float64)
    if job.get('p2clip') == 'all':
        hist[valid] = np.clip(hist[valid], CLIP_LO, CLIP_HI)
    elif job.get('p2clip') == 'current':
        hist[-1] = np.clip(hist[-1], CLIP_LO, CLIP_HI)
    elif job.get('p2clip') == 'history':
        idx = np.flatnonzero(valid)
        idx = idx[idx != len(mask) - 1]
        hist[idx] = np.clip(hist[idx], CLIP_LO, CLIP_HI)
    elif job.get('p2clip') not in (None, 'none'):
        raise ValueError('unknown p2clip ' + repr(job.get('p2clip')))
    history = hist.astype(np.float32)

    kw = policy._infer_kwargs(obs, history, mask, ns)
    steps = int(kw['num_inference_steps'])
    assert steps == STEPS_EXPECTED, 'protocol expects 10 inference steps, got %d' % steps
    proprio_norm_original = kw['proprio'].detach().float().cpu().numpy().copy()
    if job.get('p1neutral'):
        p = kw['proprio'].clone()
        p[..., 8:14] = 0
        kw['proprio'] = p
    proprio_norm_used = kw['proprio'].detach().float().cpu().numpy().copy()
    proprio_raw14 = np.asarray(policy.proprio_fn(obs), dtype=np.float32)

    random.seed(9081)
    np.random.seed(9081)
    torch.manual_seed(9081)
    torch.cuda.manual_seed_all(9081)
    trace.clear()
    hook_records.clear()
    start = time.time()
    original_randn = torch.randn
    draws = []

    def isolated_randn(*args, **kwargs):
        shape = tuple(args[0]) if len(args) == 1 and isinstance(args[0], (tuple, list)) else tuple(args)
        stream_name = 'action' if shape == (1, 32, 7) else ('force' if shape == (1, 32, 6) else 'video')
        result = original_randn(*args, **kwargs)
        draws.append({'stream': stream_name, 'shape': list(result.shape),
                      'sha256': hashlib.sha256(result.detach().float().cpu().numpy().tobytes()).hexdigest()})
        return result

    torch.randn = isolated_randn
    try:
        with torch.no_grad():
            pred = policy.model.infer_action(**copy.deepcopy(kw))
    finally:
        torch.randn = original_randn
    wall = time.time() - start
    raw = pred['action'].detach().float().cpu().numpy().reshape(32, 7)
    future = pred['force_future_newton'].detach().float().cpu().numpy()
    assert len(trace) == steps and np.isfinite(raw).all() and np.isfinite(future).all()
    assert np.array_equal(trace[-1]['after'].reshape(32, 7), raw)
    hook_summary = {k: dict(v) for k, v in hook_records.items()}

    recorded = CHUNKS['chunk_%03d_model_out' % chunk].astype(np.float64).reshape(32, 7)
    cmp = summarize_raw(raw.astype(np.float64), recorded)
    if job.get('p2clip') in (None, 'none') and not job.get('p1neutral') and not job.get('repeat'):
        a0_cache[frame] = raw.copy()
    vs_a0 = None
    if frame in a0_cache and name != 'replan_s%d_f%d_A0' % (SEED, frame):
        base = a0_cache[frame].astype(np.float64)
        cur = raw.astype(np.float64)
        vs_a0 = {
            'max_abs_delta': float(np.abs(cur - base).max()),
            'mean_abs_delta': float(np.abs(cur - base).mean()),
            'rows_changed_gt_1e-3': int((np.abs(cur - base).max(1) > 1e-3).sum()),
            'first10_dir_sign_agree': [int((np.sign(cur[:10, j]) == np.sign(base[:10, j])).sum()) for j in range(6)],
            'first10_gripper_sign_agree': int((np.sign(cur[:10, 6]) == np.sign(base[:10, 6])).sum()),
        }

    payload = {
        'raw32': raw, 'force_future_newton': future,
        'force_history_original': history_original, 'force_history_used': history,
        'force_history_mask': mask, 'normalized_proprio_original': proprio_norm_original,
        'normalized_proprio_used': proprio_norm_used, 'proprio_raw14': proprio_raw14,
        'noise_seed': np.asarray(ns),
        'trace_before': np.stack([x['before'] for x in trace]),
        'trace_after': np.stack([x['after'] for x in trace]),
        'trace_velocity': np.stack([x['velocity'] for x in trace]),
        'trace_delta': np.asarray([x['delta'] for x in trace]),
        'recorded_model_out': recorded.astype(np.float32),
        'recorded_adapter_out': CHUNKS['chunk_%03d_adapter_out' % chunk].astype(np.float32),
        **{k: v for k, v in obs.items()},
        **D.MARKERS,
    }
    np.savez_compressed(OUT / (name + '.npz'), **payload)
    rec = {
        **D.MARKERS, **job, 'chunk_index': chunk, 'num_inference_steps': steps,
        'noise_draws': draws, 'wall_seconds': wall,
        'history_before': {'min': history_original.min(0).tolist(), 'max': history_original.max(0).tolist()},
        'history_used': {'min': history.min(0).tolist(), 'max': history.max(0).tolist()},
        'history_changed_entries': int((np.abs(history.astype(np.float64) - history_original.astype(np.float64)) > 0).sum()),
        'proprio_norm_original': proprio_norm_original.reshape(-1).tolist(),
        'proprio_norm_used': proprio_norm_used.reshape(-1).tolist(),
        'proprio_raw14': proprio_raw14.tolist(),
        'compare_recorded': cmp,
        'compare_a0': vs_a0,
        'force_hooks': hook_summary,
        'observation_sha256': {k: hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() for k, v in obs.items()},
        'npz_sha256': D.sha256_file(OUT / (name + '.npz')),
    }
    write(name + '.json', rec)
    results.append(rec)
    write('summary.json', {**D.MARKERS, 'complete': False, 'results': results})
    print(json.dumps({k: rec[k] for k in ('name', 'frame', 'chunk_index', 'wall_seconds',
                                          'history_changed_entries')}, default=str)
          + ' absmax=%.4f rec=%.4f diff=%.4f'
          % (cmp['raw_absmax_32'], float(np.abs(recorded).max()), cmp['vs_recorded_full32']['max_abs_diff']),
          flush=True)


JOBS = []
FRAMES = [60, 70, 140, 150, 160, 200, 210]
for f in FRAMES:
    ns = noise_seed(SEED, f // 10 + 1)
    base = {'seed': SEED, 'frame': f, 'noise': ns}
    JOBS.append({**base, 'name': 'replan_s%d_f%d_A0' % (SEED, f), 'p2clip': 'none', 'p1neutral': False})
    JOBS.append({**base, 'name': 'replan_s%d_f%d_P2clip_all' % (SEED, f), 'p2clip': 'all', 'p1neutral': False})
    JOBS.append({**base, 'name': 'replan_s%d_f%d_P1neutral' % (SEED, f), 'p2clip': 'none', 'p1neutral': True})
    JOBS.append({**base, 'name': 'replan_s%d_f%d_P1P2' % (SEED, f), 'p2clip': 'all', 'p1neutral': True})
for f in (140, 200):
    ns = noise_seed(SEED, f // 10 + 1)
    base = {'seed': SEED, 'frame': f, 'noise': ns}
    JOBS.append({**base, 'name': 'replan_s%d_f%d_P2clip_current' % (SEED, f), 'p2clip': 'current', 'p1neutral': False})
    JOBS.append({**base, 'name': 'replan_s%d_f%d_P2clip_history' % (SEED, f), 'p2clip': 'history', 'p1neutral': False})
ns = noise_seed(SEED, 140 // 10 + 1)
JOBS.append({'seed': SEED, 'frame': 140, 'noise': ns, 'name': 'replan_s%d_f140_A0_repeat' % SEED,
             'p2clip': 'none', 'p1neutral': False, 'repeat': True})
write('predeclared_jobs.json', JOBS)

for job in JOBS:
    run(job)

write('summary.json', {**D.MARKERS, 'complete': True, 'jobs': len(results), 'results': results})
print('ALL_DONE', len(results), flush=True)
