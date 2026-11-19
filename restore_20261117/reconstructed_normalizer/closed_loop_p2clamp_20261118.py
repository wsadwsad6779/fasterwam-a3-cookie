"""Closed-loop minimal-fix test: seed17, B0 (unchanged) vs CL (P2 force history clamped).

Offline v6 proved the plan blow-up (raw32 absmax 6.75/21.25/30.50/26.75/94.0 at
f=140/150/160/200/210) is produced by the unbounded P2 force-history path: clipping every
mask-true history row to the corpus per-channel raw-newton range brings those plans back to
1.03-1.16, while neutralising the P1 force columns does not. What is still unproven is the
closed loop: does the same clamp break the press-harder -> saturate-plan -> press-harder
feedback, or is the jam already self-sustaining?

Design (single seed, simulator only):
  B0  unchanged policy -> must reproduce the recorded seed17 episode bit-for-bit; that is the
      identity gate for this script. Any nonzero diff is reported with the first diverging chunk.
  CL  the ONLY change is that `ForcePolicy._infer_kwargs` receives the sixteen-row force
      history with its mask-true rows clipped to force_calib_v1's per_channel_min/max
      (raw newtons, the same bounds v6 used). Padded rows are left untouched because the
      per-channel ranges do not contain 0.

Nothing else is modified: model, merged weights, reconstructed normalizer, pm1 rewrite,
Adapter and the eval protocol are untouched. Both arms run in one process (the model is
stateless across `start_episode`, empirically bit-exact in the A/B pair test).

Every artefact carries NORMALIZER_IDENTITY=RECONSTRUCTED_DETERMINISTIC_NOT_ORIGINAL.
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import diag_reconstructed_40k as D

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'closed_loop_p2clamp_20261118_v2'
OUT.mkdir(exist_ok=False)

FORE = ROOT / 'forensics_full_wide_out'
SEED = 17
MAX_STEPS = int(os.environ.get('CL_MAX_STEPS', '600'))
REPLAN = 10


def write(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2, default=str), encoding='utf-8')


E, factory, W, fw = D.setup_protocol()
import torch
from a3_dual_arm_sim.contracts import STATE
from force_stream import CHANNELS, CHANNEL_NAMES
import record40k as R

policy, doc, calib, facts = D.build_policy_reconstructed(
    E, W, SimpleNamespace(cpu=False, gripper_mode='pm1'))
assert facts['device'] == 'cuda' and not policy.model.training
assert list(calib['channels']) == list(CHANNELS)

CLIP_LO = np.asarray(calib['per_channel_min'], dtype=np.float64)
CLIP_HI = np.asarray(calib['per_channel_max'], dtype=np.float64)
assert CLIP_LO.shape == CLIP_HI.shape == (6,)
assert np.all(CLIP_LO < CLIP_HI)

write('identity.json', {
    **D.MARKERS,
    'scope': 'closed-loop simulator control, single seed=17, two arms in one process',
    'source_sha256': D.sha256_file(Path(__file__)),
    'protocol': facts,
    'seed': SEED, 'max_steps': MAX_STEPS, 'replan': REPLAN,
    'clip_bounds_newton': {'lo': CLIP_LO.tolist(), 'hi': CLIP_HI.tolist(),
                           'channel_names': list(CHANNEL_NAMES),
                           'source': 'step_040000/force_calib_v1.json per_channel_min/max'},
    'arms': {
        'B0': 'no change; must reproduce forensics_full_wide_out seed17 bit-exactly',
        'CL': 'clip every mask-true force-history row to the corpus per-channel raw range '
              'inside _infer_kwargs only; padded rows untouched',
    },
    'untouched': ['model', 'merged weights', 'reconstructed normalizer', 'pm1 rewrite',
                  'Adapter', 'eval protocol files', 'forensics outputs'],
})


def snapshot_step(store, obs, info):
    store['eef_pose'].append(np.asarray(obs['observation.eef_pose'], dtype=np.float64))
    store['force18'].append(np.asarray(obs['observation.force'], dtype=np.float64))
    store['opening'].append(float(np.asarray(obs[STATE])[7]))
    store['cookies'].append(info.get('cookies_in_target'))
    store['success'].append(bool(info.get('success'))
                            if info.get('success') is not None else None)


def run_arm(tag, clamp):
    env = factory.make_env(position_noise_m=0.0, yaw_noise_rad=0.0, render_cameras=True)
    reset = D.make_reset(E, factory, env, SimpleNamespace(
        cpu=False, gripper_mode='pm1', max_steps=MAX_STEPS, replan=REPLAN,
        allow_invalid_scene=False))
    store = {k: [] for k in ('eef_pose', 'force18', 'opening', 'cookies', 'success')}
    clip_log = []

    real_step = env.step

    def step(action, **k):
        outv = real_step(action, **k)
        snapshot_step(store, outv[0], outv[4] if len(outv) > 4 else {})
        return outv

    env.step = step

    def reset_and_snapshot(env_, s):
        obs = reset(env_, s)
        snapshot_step(store, obs, {})
        return obs

    real_infer_kwargs = policy._infer_kwargs
    if clamp:
        def clamped(obs, history, mask, ns):
            h = np.array(history, dtype=np.float64, copy=True)
            m = np.asarray(mask).astype(bool)
            before = h[m] if m.any() else h[:0]
            if m.any():
                h[m] = np.clip(h[m], CLIP_LO, CLIP_HI)
            after = h[m] if m.any() else h[:0]
            clip_log.append({
                'mask_true': int(m.sum()),
                'entries_clipped': int(np.count_nonzero(np.abs(after - before) > 0)),
                'history_min_before': before.min(0).tolist() if before.size else None,
                'history_max_before': before.max(0).tolist() if before.size else None,
            })
            return real_infer_kwargs(obs, h.astype(np.asarray(history).dtype), mask, ns)
        policy._infer_kwargs = clamped

    real_bridge, real_convert = policy.bridge, policy.adapter.convert
    spy = R.ActionSpy().install(policy)
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    t0 = time.time()
    try:
        result = E.run_episode(env, policy, seed=SEED, reset=reset_and_snapshot,
                               max_steps=MAX_STEPS, replan=REPLAN)
    finally:
        policy.bridge, policy.adapter.convert = real_bridge, real_convert
        policy._infer_kwargs = real_infer_kwargs
        env.step = real_step
    wall = round(time.time() - t0, 2)
    rec, measured, chunks = D.serialize_result(result)
    if hasattr(env, 'close'):
        env.close()

    payload = {'measured': measured,
               'chunk_pred_force': np.asarray([c['pred_force'] if c['pred_force'] is not None
                                               else np.full((32, 6), np.nan) for c in chunks],
                                              dtype=np.float64),
               'chunk_frames': np.asarray([c['frame'] for c in chunks], dtype=np.int64),
               'chunk_meta': np.asarray([json.dumps(c['meta'], default=str) for c in chunks]),
               'step_eef_pose': np.asarray(store['eef_pose'], dtype=np.float64),
               'step_force18': np.asarray(store['force18'], dtype=np.float64),
               'step_opening': np.asarray(store['opening'], dtype=np.float64),
               'step_cookies': np.asarray([-1 if c is None else c for c in store['cookies']],
                                          dtype=np.int64),
               'step_success': np.asarray([-1 if s is None else int(s) for s in store['success']],
                                          dtype=np.int64),
               'clip_log': np.asarray([json.dumps(x, sort_keys=True) for x in clip_log]),
               'infer_calls': np.asarray(int(policy.calls)),
               **{('chunk_%03d_%s' % (i, key)): np.asarray(value)
                  for i, row in enumerate(spy.rows) for key, value in row.items()},
               **D.MARKERS}
    np.savez_compressed(OUT / (tag + '.npz'), **payload)

    m = measured.astype(np.float64)
    eef = np.asarray(store['eef_pose'], dtype=np.float64)
    steps = max(1, eef.shape[0] - 1)
    disp = np.linalg.norm(eef[1:, :3] - eef[:-1, :3], axis=1) * 1000.0
    rec_out = {
        **D.MARKERS, 'arm': tag, 'clamp': bool(clamp), 'seed': SEED,
        'max_steps': MAX_STEPS, 'replan': REPLAN, 'record': rec, 'chunks': chunks,
        'infer_calls': int(policy.calls), 'wall_s': wall,
        'force_envelope_newton': {'min': m.min(0).tolist(), 'max': m.max(0).tolist(),
                                  'absmax': float(np.abs(m).max()),
                                  'frames_absmax_gt_14_3': int((np.abs(m) > 14.3).any(1).sum()),
                                  'frames_absmax_gt_50': int((np.abs(m) > 50.0).any(1).sum())},
        'eef_step_mm': {'mean': float(disp.mean()), 'median': float(np.median(disp)),
                        'max': float(disp.max()), 'p90': float(np.percentile(disp, 90))},
        'clip_log_entries': len(clip_log),
        'clip_log_first20': clip_log[:20],
        'npz_sha256': D.sha256_file(OUT / (tag + '.npz')),
    }
    write(tag + '.json', rec_out)
    print('[arm %s] wall=%.1fs infer_calls=%d absmax=%.3f gt14.3=%d cookies=%s success=%s'
          % (tag, wall, policy.calls, rec_out['force_envelope_newton']['absmax'],
             rec_out['force_envelope_newton']['frames_absmax_gt_14_3'],
             rec['cookies_in_target'], rec['success']), flush=True)
    return {'tag': tag, 'record': rec, 'measured': measured, 'spy': spy, 'store': store,
            'clip_log': clip_log, 'npz': tag + '.npz'}


def identity_check(b0):
    """B0 vs the archived seed17 forensics episode: plans, adapter rows, measured forces."""
    rec_chunks = np.load(FORE / 'seed17_original_chunks.npz', allow_pickle=False)
    rec_meas = np.load(FORE / 'seed17_actions.npz', allow_pickle=False)['measured']
    chunks = []
    for i, row in enumerate(b0['spy'].rows):
        key = 'chunk_%03d_model_out' % i
        if key not in rec_chunks.files:
            break
        a = np.asarray(row['model_out'], dtype=np.float64).reshape(32, 7)
        b = rec_chunks[key].astype(np.float64).reshape(32, 7)
        chunks.append({'chunk': i, 'max_abs_diff': float(np.abs(a - b).max()),
                       'bit_equal': bool(np.array_equal(a, b))})
    meas = np.asarray(b0['measured'], dtype=np.float64)
    n = min(len(meas), len(rec_meas))
    same_shape = meas.shape == np.asarray(rec_meas).shape
    meas_diff = float(np.abs(meas[:n] - np.asarray(rec_meas, dtype=np.float64)[:n]).max()) \
        if n else None
    first_bad = next((c['chunk'] for c in chunks if not c['bit_equal']), None)
    out = {'chunks_compared': len(chunks),
           'chunks_bit_equal': int(sum(c['bit_equal'] for c in chunks)),
           'first_divergent_chunk': first_bad,
           'model_out_max_abs_diff': max((c['max_abs_diff'] for c in chunks), default=None),
           'measured_shape_equal': bool(same_shape),
           'measured_rows_compared': n,
           'measured_max_abs_diff': meas_diff,
           'gate_passed': bool(chunks and all(c['bit_equal'] for c in chunks)
                               and same_shape and meas_diff == 0.0)}
    write('identity_gate.json', {**D.MARKERS, 'seed': SEED,
                                 'reference': 'forensics_full_wide_out seed17 (600 steps)',
                                 **out, 'per_chunk': chunks})
    return out


def compare(b0, cl):
    mb = np.asarray(b0['measured'], dtype=np.float64)
    mc = np.asarray(cl['measured'], dtype=np.float64)
    rb, rc = b0['record'], cl['record']
    first_div = None
    n = min(len(mb), len(mc))
    for f in range(n):
        if not np.array_equal(mb[f], mc[f]):
            first_div = f
            break
    eb = np.asarray(b0['store']['eef_pose'], dtype=np.float64)
    ec = np.asarray(cl['store']['eef_pose'], dtype=np.float64)
    out = {
        'B0_record': rb, 'CL_record': rc,
        'first_force_divergence_frame': first_div,
        'measured_rows_B0': len(mb), 'measured_rows_CL': len(mc),
        'B0_absmax': float(np.abs(mb).max()), 'CL_absmax': float(np.abs(mc).max()),
        'B0_frames_gt_50N': int((np.abs(mb) > 50).any(1).sum()),
        'CL_frames_gt_50N': int((np.abs(mc) > 50).any(1).sum()),
        'B0_frames_gt_14_3N': int((np.abs(mb) > 14.3).any(1).sum()),
        'CL_frames_gt_14_3N': int((np.abs(mc) > 14.3).any(1).sum()),
        'B0_final_eef': eb[-1, :3].tolist() if len(eb) else None,
        'CL_final_eef': ec[-1, :3].tolist() if len(ec) else None,
        'clip_calls': len(cl['clip_log']),
        'clip_calls_with_changes': int(sum(1 for x in cl['clip_log'] if x['entries_clipped'])),
    }
    write('comparison.json', {**D.MARKERS, 'seed': SEED, **out})
    return out


write('predeclared.json', {**D.MARKERS, 'arms': ['B0', 'CL'], 'seed': SEED,
                           'max_steps': MAX_STEPS, 'replan': REPLAN,
                           'clip_bounds_newton': {'lo': CLIP_LO.tolist(), 'hi': CLIP_HI.tolist()}})

b0 = run_arm('B0', clamp=False)
gate = identity_check(b0)
print('[identity] ' + json.dumps({k: v for k, v in gate.items() if k != 'per_chunk'}), flush=True)

cl = run_arm('CL', clamp=True)
cmp = compare(b0, cl)
print('[compare] ' + json.dumps(cmp), flush=True)
write('summary.json', {**D.MARKERS, 'complete': True, 'seed': SEED,
                       'max_steps': MAX_STEPS,
                       'identity_gate': {k: v for k, v in gate.items() if k != 'per_chunk'},
                       'comparison': cmp})
print('ALL_DONE', flush=True)
