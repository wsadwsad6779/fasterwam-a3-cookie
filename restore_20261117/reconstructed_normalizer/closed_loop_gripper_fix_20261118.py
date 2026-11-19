"""Closed-loop minimal-fix test for the gripper chatter.

Evidence this tests
-------------------
prepare_a3_arrays.py derives action dim 7 as sign(measured opening[t+1]-opening[t]) with a
1e-4 deadband and hold-previous, so sub-0.1% settling jitter becomes sign changes: 57% of all
training-label flips have |d opening| < 1e-3, and demo windows at grasp/release states carry
3-8 flips per 32 frames while the expert's commanded joint target is constant. The policy
reproduces that propensity at matched states (f0 5.8 vs 4.75, f30 3.0 vs 3.05, f110 4.2 vs
3.25) and runs hotter where it extrapolates (f440 14.2 vs 7.25).

This script tests two deployment-side post-processors on the predicted dim 7, on top of the
P2 force-history clamp that already removed the jam:

  CL   reference: P2 clamp only. Must reproduce the stored v2 CL arm bit exactly.
  M5   P2 clamp + temporal median filter (k=5, edge padded) on the predicted dim 7.
  H8   P2 clamp + hysteresis (threshold 0.8 of the normalised range, hold previous sign).

Metrics: executed gripper sign flips, short open/close pulses, close-run -> open transitions
(necessary releases must survive), force envelope, jam steps, cookies in source and target.
Scene state is logged every step (cookie bodies) as in closed_loop_scene_20261118.py.

Only the predicted action is post-processed inside this process; model, weights, normalizer,
pm1, Adapter and protocol files are untouched, and no counterfactual is deployed.
"""
from __future__ import annotations

import base64
import copy
import functools
import json
import os
import random
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import diag_reconstructed_40k as D

ROOT = Path(__file__).resolve().parent
OUT = ROOT / os.environ.get('CL_FIX_OUT', 'closed_loop_gripper_fix_20261118')
OUT.mkdir(exist_ok=False)
V2 = ROOT / 'closed_loop_p2clamp_20261118_v2'
SEED = int(os.environ.get('CL_FIX_SEED', '17'))
MAX_STEPS = int(os.environ.get('CL_MAX_STEPS', '600'))
REPLAN = 10
ARMS = os.environ.get('CL_FIX_ARMS', 'CL,M5,H8').split(',')


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

write('identity.json', {
    **D.MARKERS,
    'scope': 'closed-loop simulator, seed 17, P2 clamp + predicted-dim7 post-processors',
    'source_sha256': D.sha256_file(Path(__file__)),
    'protocol': facts, 'seed': SEED, 'max_steps': MAX_STEPS, 'replan': REPLAN,
    'arms': {'CL': 'P2 clamp only (reference, must match closed_loop_p2clamp_20261118_v2/CL)',
             'M5': 'P2 clamp + median filter k=5 on predicted dim 7',
             'H8': 'P2 clamp + hysteresis 0.8 on predicted dim 7'},
    'untouched': ['model', 'weights', 'reconstructed normalizer', 'pm1', 'Adapter',
                  'protocol files', 'recorded artefacts'],
})


def med_filter(x, k=5):
    pad = k // 2
    y = np.pad(x, (pad, pad), mode='edge')
    return np.array([np.median(y[i:i + k]) for i in range(len(x))])


def hyst(x, h=0.8, prev=1.0):
    out = np.empty_like(x)
    s = prev
    for t, v in enumerate(x):
        if abs(v) >= h:
            s = 1.0 if v >= 0 else -1.0
        out[t] = s
    return out


def runs(mask):
    out, i = [], 0
    while i < len(mask):
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < len(mask) and mask[j]:
            j += 1
        out.append((i, j - 1, j - i))
        i = j
    return out


def run_arm(tag):
    fix = None if tag == 'CL' else ('M5' if tag == 'M5' else 'H8')
    env = factory.make_env(position_noise_m=0.0, yaw_noise_rad=0.0, render_cameras=True)
    reset = D.make_reset(E, factory, env, SimpleNamespace(
        cpu=False, gripper_mode='pm1', max_steps=MAX_STEPS, replan=REPLAN,
        allow_invalid_scene=False))
    keys = ('eef_pose', 'force18', 'opening', 'cookies', 'success', 'cookie_pos',
            'cookies_in_source')
    store = {k: [] for k in keys}
    clip_log = []

    def scene(env_, info):
        cp = info.get('cookie_positions')
        if cp is None:
            for obj in (env_, getattr(env_, 'unwrapped', None)):
                if obj is not None and hasattr(obj, 'cookie_positions'):
                    try:
                        cp = obj.cookie_positions
                        break
                    except Exception:
                        cp = None
        prev = next((p for p in reversed(store['cookie_pos']) if p.size), None)
        if cp is None:
            store['cookie_pos'].append(np.full(prev.shape, np.nan, np.float32)
                                       if prev is not None else np.zeros((0, 3), np.float32))
        else:
            store['cookie_pos'].append(np.asarray(cp, dtype=np.float32).reshape(-1, 3))
        store['cookies_in_source'].append(int(info.get('cookies_in_source', -1)))

    def snap(obs, info, env_):
        store['eef_pose'].append(np.asarray(obs['observation.eef_pose'], dtype=np.float64))
        store['force18'].append(np.asarray(obs['observation.force'], dtype=np.float64))
        store['opening'].append(float(np.asarray(obs[STATE])[7]))
        store['cookies'].append(info.get('cookies_in_target'))
        store['success'].append(bool(info.get('success'))
                                if info.get('success') is not None else None)
        scene(env_, info)

    real_step = env.step

    def step(action, **k):
        outv = real_step(action, **k)
        snap(outv[0], outv[4] if len(outv) > 4 else {}, env)
        return outv

    env.step = step

    def reset_and_snapshot(env_, s):
        obs = reset(env_, s)
        snap(obs, {}, env_)
        return obs

    real_infer_kwargs = policy._infer_kwargs

    def clamped(obs, history, mask, ns):
        h = np.array(history, dtype=np.float64, copy=True)
        m = np.asarray(mask).astype(bool)
        before = h[m] if m.any() else h[:0]
        if m.any():
            h[m] = np.clip(h[m], CLIP_LO, CLIP_HI)
        after = h[m] if m.any() else h[:0]
        clip_log.append({'mask_true': int(m.sum()),
                         'entries_clipped': int(np.count_nonzero(np.abs(after - before) > 0))})
        return real_infer_kwargs(obs, h.astype(np.asarray(history).dtype), mask, ns)

    policy._infer_kwargs = clamped

    real_infer = policy.model.infer_action
    if fix is not None:
        @functools.wraps(real_infer)
        def patched_infer(*args, **kwargs):
            pred = real_infer(*args, **kwargs)
            a = pred['action']
            local = copy.copy(pred)
            vt = a[..., 6]
            v = vt.detach().float().cpu().numpy().reshape(-1)
            v2 = med_filter(v, 5) if fix == 'M5' else hyst(v, 0.8, prev=1.0)
            b = a.clone()
            b[..., 6] = torch.as_tensor(v2, dtype=a.dtype, device=a.device).reshape(vt.shape)
            local['action'] = b
            return local
        policy.model.infer_action = patched_infer

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
        policy.model.infer_action = real_infer
        env.step = real_step
    wall = round(time.time() - t0, 2)
    rec, measured, chunks = D.serialize_result(result)
    if hasattr(env, 'close'):
        env.close()

    cp = store['cookie_pos']
    shapes = sorted({p.shape for p in cp})
    if len(shapes) > 1:
        good = max(shapes, key=lambda s: s[0])
        cp = [p if p.shape == good else np.full(good, np.nan, np.float32) for p in cp]

    grip = np.zeros(len(store['opening']))
    for k, row in enumerate(spy.rows):
        ad = np.asarray(row.get('adapter_out'), dtype=np.float64).reshape(-1, 14) \
            if row.get('adapter_out') is not None else None
        if ad is None:
            continue
        m = min(10, len(grip) - 10 * k)
        if m > 0:
            grip[10 * k:10 * k + m] = ad[:m, 6]

    payload = {
        'measured': measured,
        'chunk_pred_force': np.asarray([c['pred_force'] if c['pred_force'] is not None
                                        else np.full((32, 6), np.nan) for c in chunks], np.float64),
        'chunk_meta': np.asarray([json.dumps(c['meta'], default=str) for c in chunks]),
        'step_eef_pose': np.asarray(store['eef_pose'], np.float64),
        'step_force18': np.asarray(store['force18'], np.float64),
        'step_opening': np.asarray(store['opening'], np.float64),
        'step_cookies': np.asarray([-1 if c is None else c for c in store['cookies']], np.int64),
        'step_success': np.asarray([-1 if s is None else int(s) for s in store['success']], np.int64),
        'step_cookie_pos': np.asarray(cp, np.float32),
        'step_cookies_in_source': np.asarray(store['cookies_in_source'], np.int64),
        'step_grip_cmd': grip,
        'clip_log': np.asarray([json.dumps(x, sort_keys=True) for x in clip_log]),
        **{('chunk_%03d_%s' % (i, key)): np.asarray(value)
           for i, row in enumerate(spy.rows) for key, value in row.items()},
        **D.MARKERS,
    }
    np.savez_compressed(OUT / (tag + '.npz'), **payload)

    sg = np.where(grip >= 0, 1, -1)
    flips = int((np.diff(sg) != 0).sum())
    closed = sg < 0
    pulse = [l for (s, e, l) in runs(closed) if l <= 2]
    releases = 0
    for (s, e, l) in runs(closed):
        if l >= 10 and e + 1 < len(sg) and sg[e + 1] > 0:
            releases += 1
    m = measured.astype(np.float64)
    out = {**D.MARKERS, 'arm': tag, 'seed': SEED, 'wall_s': wall, 'record': rec,
           'executed_grip_flips': flips,
           'executed_closed_fraction': float(closed.mean()),
           'short_close_pulses_le2': len(pulse),
           'close_runs_ge10_followed_by_open': releases,
           'force_absmax': float(np.abs(m).max()),
           'cookies_in_target': rec['cookies_in_target'],
           'cookies_in_source_final': next((c for c in reversed(store['cookies_in_source'])
                                            if c >= 0), -1),
           'npz_sha256': D.sha256_file(OUT / (tag + '.npz'))}
    write(tag + '.json', out)
    print('[%s] wall=%.1fs flips=%d short_pulses=%d releases=%d force=%.1f cookies=%s src=%s'
          % (tag, wall, flips, len(pulse), releases, out['force_absmax'],
             rec['cookies_in_target'], out['cookies_in_source_final']), flush=True)
    return out


write('predeclared.json', {**D.MARKERS, 'seed': SEED, 'arms': ARMS, 'max_steps': MAX_STEPS,
                           'replan': REPLAN,
                           'hypothesis': 'dim7 chatter is inherited from the label construction; '
                                         'a temporal median (M5) should cut executed flips toward '
                                         'the supervision rate while keeping every real release, '
                                         'whereas a strong hysteresis (H8) may suppress transitions'})
results = []
for tag in ARMS:
    try:
        results.append(run_arm(tag))
    except Exception as exc:
        results.append({'arm': tag, 'error': type(exc).__name__ + ': %s' % exc})
        print('[%s] failed: %s' % (tag, exc), flush=True)
    write('summary.json', {**D.MARKERS, 'complete': False, 'results': results})

# reproducibility gate for the CL arm against the stored v2 CL arm
gate = None
if (OUT / 'CL.npz').exists() and (V2 / 'CL.npz').exists():
    a = np.load(OUT / 'CL.npz')
    b = np.load(V2 / 'CL.npz')
    parts = {}
    for k in ('step_eef_pose', 'step_force18', 'step_opening'):
        x = np.asarray(a[k], np.float64)
        y = np.asarray(b[k], np.float64)
        n = min(len(x), len(y))
        parts[k] = {'bit_equal': bool(x.shape == y.shape and np.array_equal(x, y)),
                    'max_abs_diff': float(np.abs(x[:n] - y[:n]).max()) if n else None}
    gate = {'vs': 'closed_loop_p2clamp_20261118_v2/CL.npz', **parts,
            'passed': all(v['bit_equal'] for v in parts.values())}
    write('repro_gate_cl.json', {**D.MARKERS, **gate})
    print('[gate]', json.dumps(gate), flush=True)

write('summary.json', {**D.MARKERS, 'complete': True, 'seed': SEED, 'arms': ARMS,
                       'repro_gate_cl': gate, 'results': results})
print('ALL_DONE', flush=True)