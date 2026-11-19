"""v3 scene-truth closed loop: does the P2-clamped policy actually grasp and deliver cookies?

v2 showed the P2 clamp removes the jam but leaves cookies_in_target = 0 with two
pinch-carry-release cycles. v2 logged no object state, so "0" cannot be attributed to
either an empty grasp or lost plates. This run logs simulator cookie bodies every step.

Arms (predeclared): CL (P2 clip) on seeds 17, 11, 21. No B0 arm: the seed17 B0 baseline and
its bit-exact identity gate already exist in closed_loop_p2clamp_20261118_v2.
Seed 17 doubles as a reproducibility gate: its step arrays must match the stored v2 CL arm
bit-for-bit, proving the extra logging is passive.

Everything else is unchanged (model, weights, reconstructed normalizer, pm1, Adapter, protocol).
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
OUT = ROOT / 'closed_loop_scene_20261118_v2'
OUT.mkdir(exist_ok=False)

V2 = ROOT / 'closed_loop_p2clamp_20261118_v2'
MAX_STEPS = int(os.environ.get('CL_MAX_STEPS', '600'))
REPLAN = 10
SEEDS = [int(s) for s in os.environ.get('CL_SEEDS', '17,11,21').split(',') if s.strip()]


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
assert CLIP_LO.shape == CLIP_HI.shape == (6,) and np.all(CLIP_LO < CLIP_HI)

write('identity.json', {
    **D.MARKERS,
    'scope': 'closed-loop simulator control, P2-clamped arm only, seeds 17/11/21',
    'source_sha256': D.sha256_file(Path(__file__)),
    'protocol': facts,
    'seeds': SEEDS, 'max_steps': MAX_STEPS, 'replan': REPLAN,
    'clip_bounds_newton': {'lo': CLIP_LO.tolist(), 'hi': CLIP_HI.tolist(),
                           'channel_names': list(CHANNEL_NAMES)},
    'new_logging': ['cookie_positions (all cookie bodies, every step)',
                    'cookies_in_target / cookies_in_source / in-target and in-source masks',
                    'target_slot_occupancy', 'target_touches_all_walls',
                    'full observation.state (14)'],
    'untouched': ['model', 'merged weights', 'reconstructed normalizer', 'pm1 rewrite',
                  'Adapter', 'eval protocol files', 'forensics outputs', 'v2 artefacts'],
})

N_COOKIE = None
COOKIE_NOTE = None


def scene_snapshot(store, env, info):
    cp = info.get('cookie_positions')
    if cp is None:
        for obj in (env, getattr(env, 'unwrapped', None)):
            if obj is not None and hasattr(obj, 'cookie_positions'):
                try:
                    cp = obj.cookie_positions
                    break
                except Exception:
                    cp = None
    global N_COOKIE
    if cp is None:
        prev = next((p for p in reversed(store['cookie_pos']) if p.size), None)
        store['cookie_pos'].append(np.full(prev.shape, np.nan, dtype=np.float32)
                                   if prev is not None else np.zeros((0, 3), np.float32))
    else:
        arr = np.asarray(cp, dtype=np.float32).reshape(-1, 3)
        if N_COOKIE is None:
            N_COOKIE = arr.shape[0]
        store['cookie_pos'].append(arr)
    store['in_target_mask'].append(np.asarray(info.get('cookies_in_target_mask', []), dtype=bool))
    store['in_source_mask'].append(np.asarray(info.get('cookies_in_source_mask', []), dtype=bool))
    store['slot_occupancy'].append(np.asarray(info.get('target_slot_occupancy', []), dtype=np.int16))
    store['touches_walls'].append(bool(info.get('target_touches_all_walls', False)))
    store['cookies_in_source'].append(int(info.get('cookies_in_source', -1)))


def snapshot_step(store, obs, info, env):
    store['eef_pose'].append(np.asarray(obs['observation.eef_pose'], dtype=np.float64))
    store['force18'].append(np.asarray(obs['observation.force'], dtype=np.float64))
    store['state14'].append(np.asarray(obs[STATE], dtype=np.float64))
    store['opening'].append(float(np.asarray(obs[STATE])[7]))
    store['cookies'].append(info.get('cookies_in_target'))
    store['success'].append(bool(info.get('success'))
                            if info.get('success') is not None else None)
    scene_snapshot(store, env, info)


def _fit_rows(rows, fill, dtype):
    """Pad/trim ragged per-step vectors (the reset frame has no info) to one width."""
    width = max((len(r) for r in rows), default=0)
    out = np.full((len(rows), width), fill, dtype=dtype)
    flags = np.zeros(len(rows), dtype=bool)
    for i, r in enumerate(rows):
        if len(r) == width and width:
            out[i] = r
        elif len(r) == 0:
            flags[i] = True
    return out, flags


def run_arm(seed):
    tag = 'CL_s%d' % seed
    env = factory.make_env(position_noise_m=0.0, yaw_noise_rad=0.0, render_cameras=True)
    reset = D.make_reset(E, factory, env, SimpleNamespace(
        cpu=False, gripper_mode='pm1', max_steps=MAX_STEPS, replan=REPLAN,
        allow_invalid_scene=False))
    keys = ('eef_pose', 'force18', 'state14', 'opening', 'cookies', 'success',
            'cookie_pos', 'in_target_mask', 'in_source_mask', 'slot_occupancy',
            'touches_walls', 'cookies_in_source')
    store = {k: [] for k in keys}
    clip_log = []

    real_step = env.step

    def step(action, **k):
        outv = real_step(action, **k)
        snapshot_step(store, outv[0], outv[4] if len(outv) > 4 else {}, env)
        return outv

    env.step = step

    def reset_and_snapshot(env_, s):
        obs = reset(env_, s)
        snapshot_step(store, obs, {}, env_)
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
                         'entries_clipped': int(np.count_nonzero(np.abs(after - before) > 0)),
                         'history_min_before': before.min(0).tolist() if before.size else None,
                         'history_max_before': before.max(0).tolist() if before.size else None})
        return real_infer_kwargs(obs, h.astype(np.asarray(history).dtype), mask, ns)

    policy._infer_kwargs = clamped

    real_bridge, real_convert = policy.bridge, policy.adapter.convert
    spy = R.ActionSpy().install(policy)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    t0 = time.time()
    try:
        result = E.run_episode(env, policy, seed=seed, reset=reset_and_snapshot,
                               max_steps=MAX_STEPS, replan=REPLAN)
    finally:
        policy.bridge, policy.adapter.convert = real_bridge, real_convert
        policy._infer_kwargs = real_infer_kwargs
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
    in_t, miss_t = _fit_rows(store['in_target_mask'], False, bool)
    in_s, miss_s = _fit_rows(store['in_source_mask'], True, bool)
    slots, miss_slot = _fit_rows(store['slot_occupancy'], -1, np.int16)
    scene_missing = miss_t | miss_s | miss_slot

    payload = {
        'measured': measured,
        'chunk_pred_force': np.asarray([c['pred_force'] if c['pred_force'] is not None
                                        else np.full((32, 6), np.nan) for c in chunks], np.float64),
        'chunk_frames': np.asarray([c['frame'] for c in chunks], dtype=np.int64),
        'chunk_meta': np.asarray([json.dumps(c['meta'], default=str) for c in chunks]),
        'step_eef_pose': np.asarray(store['eef_pose'], np.float64),
        'step_force18': np.asarray(store['force18'], np.float64),
        'step_state14': np.asarray(store['state14'], np.float64),
        'step_opening': np.asarray(store['opening'], np.float64),
        'step_cookies': np.asarray([-1 if c is None else c for c in store['cookies']], np.int64),
        'step_success': np.asarray([-1 if s is None else int(s) for s in store['success']], np.int64),
        'step_cookie_pos': np.asarray(cp, np.float32),
        'step_in_target_mask': in_t,
        'step_in_source_mask': in_s,
        'step_slot_occupancy': slots,
        'step_scene_info_missing': scene_missing,
        'step_touches_walls': np.asarray(store['touches_walls'], bool),
        'step_cookies_in_source': np.asarray(store['cookies_in_source'], np.int64),
        'clip_log': np.asarray([json.dumps(x, sort_keys=True) for x in clip_log]),
        'infer_calls': np.asarray(int(policy.calls)),
        **{('chunk_%03d_%s' % (i, key)): np.asarray(value)
           for i, row in enumerate(spy.rows) for key, value in row.items()},
        **D.MARKERS,
    }
    np.savez_compressed(OUT / (tag + '.npz'), **payload)

    eef = np.asarray(store['eef_pose'], np.float64)
    step_mm = np.linalg.norm(eef[1:, :3] - eef[:-1, :3], axis=1) * 1000.0
    cp_arr = np.asarray(cp, np.float32) if len(cp) else np.zeros((0, 0, 3), np.float32)
    ck = [-1 if c is None else int(c) for c in store['cookies']]
    rec_out = {
        **D.MARKERS, 'arm': tag, 'seed': seed, 'clamp': 'P2 all mask-true rows',
        'max_steps': MAX_STEPS, 'replan': REPLAN, 'record': rec, 'chunks': chunks,
        'infer_calls': int(policy.calls), 'wall_s': wall,
        'cookie_count': int(cp_arr.shape[1]) if cp_arr.ndim == 3 and cp_arr.size else 0,
        'cookies_in_target_trace': {'first_nonzero': int(np.argmax(np.asarray(ck) > 0))
                                    if any(c > 0 for c in ck) else None,
                                    'final': rec['cookies_in_target'], 'max': max(ck)},
        'cookies_in_source_trace': {'start': next((c for c in store['cookies_in_source'] if c >= 0), -1),
                                    'final': store['cookies_in_source'][-1],
                                    'min': min([c for c in store['cookies_in_source'] if c >= 0], default=-1)},
        'slot_occupancy_final': np.asarray(store['slot_occupancy'][-1]).tolist()
        if len(store['slot_occupancy']) else None,
        'touches_walls_any': bool(np.asarray(store['touches_walls'], bool).any())
        if len(store['touches_walls']) else None,
        'eef_step_mm': {'mean': float(step_mm.mean()), 'median': float(np.median(step_mm)),
                        'max': float(step_mm.max())},
        'eef_start': eef[0, :3].tolist() if len(eef) else None,
        'eef_end': eef[-1, :3].tolist() if len(eef) else None,
        'clip_log_entries': len(clip_log),
        'npz_sha256': D.sha256_file(OUT / (tag + '.npz')),
    }
    write(tag + '.json', rec_out)
    print('[%s] wall=%.1fs cookies=%s success=%s eef_end=%s'
          % (tag, wall, rec['cookies_in_target'], rec['success'],
             np.round(eef[-1, :3], 4).tolist() if len(eef) else None), flush=True)
    return {'tag': tag, 'seed': seed, 'record': rec, 'summary': rec_out, 'npz': tag + '.npz',
            'store': store, 'clip_log': clip_log}


def repro_gate(arm):
    """Seed 17 must reproduce the stored v2 CL arm bit-exactly."""
    v2 = np.load(str(V2 / 'CL.npz'), allow_pickle=False)
    out = {}
    for key in ('step_eef_pose', 'step_force18', 'step_opening'):
        a = np.asarray(arm['store'][{'step_eef_pose': 'eef_pose', 'step_force18': 'force18',
                                     'step_opening': 'opening'}[key]], np.float64)
        b = np.asarray(v2[key], np.float64)
        n = min(len(a), len(b))
        same = a.shape == b.shape
        out[key] = {'shape_equal': bool(same), 'rows': n,
                    'max_abs_diff': float(np.abs(a[:n] - b[:n]).max()) if n else None,
                    'bit_equal': bool(same and np.array_equal(a, b))}
    out['passed'] = all(v['bit_equal'] for v in out.values())
    return out


write('predeclared.json', {**D.MARKERS, 'arms': ['CL_s%d' % s for s in SEEDS],
                           'seed': SEEDS, 'max_steps': MAX_STEPS, 'replan': REPLAN,
                           'hypothesis': 'CL reaches the bin footprint but the plates are never held; '
                                         'cookie body positions will show whether any cookie moves with '
                                         'the gripper and where plates end up'})

results = []
for seed in SEEDS:
    try:
        arm = run_arm(seed)
        rec = {'tag': arm['tag'], 'seed': seed, 'wall_s': arm['summary'].get('wall_s'),
               'cookies_in_target': arm['record']['cookies_in_target'],
               'success': arm['record']['success'],
               'cookies_in_source_trace': arm['summary']['cookies_in_source_trace']}
        if seed == 17 and MAX_STEPS >= 600:
            rec['repro_gate_vs_v2_CL'] = repro_gate(arm)
            write('repro_gate_seed17.json', {**D.MARKERS, **rec['repro_gate_vs_v2_CL']})
            print('[repro] ' + json.dumps(rec['repro_gate_vs_v2_CL']), flush=True)
        results.append(rec)
    except (SystemExit, Exception) as exc:
        results.append({'seed': seed, 'error': type(exc).__name__ + ': %s' % exc})
        print('[seed %d] aborted: %s' % (seed, exc), flush=True)
    write('summary.json', {**D.MARKERS, 'complete': False, 'results': results})

write('summary.json', {**D.MARKERS, 'complete': True, 'seeds': SEEDS, 'results': results})
print('ALL_DONE', flush=True)