"""Placement decomposition probe: why does a plate that sits in the bin not count?

Runs one P2-clamped episode (seed configurable), logs cookie body rotations every step, and at
the end recomputes the sim's own in-target test term by term for every cookie:

  p_rel, half_extent = |R_rel| @ COOKIE_HALF_SIZE
  footprint_inside = all(|p_rel[:2]| + half_extent[:2] <= TARGET_INNER_HALF_SIZE + tol)
  bottom = p_rel[2] - half_extent[2]
  vertically_inside = (bottom >= floor_top - 0.003) and (bottom < TARGET_WALL_HEIGHT)
  released          = bottom <= floor_top + 0.004
  in_target         = footprint_inside and vertically_inside [and released]

Constants are read from the live env, not hard-coded. No asset is modified.
"""
from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import diag_reconstructed_40k as D

ROOT = Path(__file__).resolve().parent
SEED = int(os.environ.get('CL_PROBE_SEED', '17'))
MAX_STEPS = int(os.environ.get('CL_MAX_STEPS', '600'))
REPLAN = 10
OUT = ROOT / os.environ.get('CL_PROBE_OUT', 'closed_loop_placement_probe_20261118')
OUT.mkdir(exist_ok=False)


def write(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2, default=str), encoding='utf-8')


E, factory, W, fw = D.setup_protocol()
import torch
from a3_dual_arm_sim.contracts import STATE
from force_stream import CHANNELS, CHANNEL_NAMES

policy, doc, calib, facts = D.build_policy_reconstructed(
    E, W, SimpleNamespace(cpu=False, gripper_mode='pm1'))
assert facts['device'] == 'cuda' and not policy.model.training

CLIP_LO = np.asarray(calib['per_channel_min'], dtype=np.float64)
CLIP_HI = np.asarray(calib['per_channel_max'], dtype=np.float64)
write('identity.json', {**D.MARKERS, 'source_sha256': D.sha256_file(Path(__file__)),
                        'protocol': facts, 'seed': SEED, 'max_steps': MAX_STEPS,
                        'scope': 'P2 clamp + cookie orientation logging; no control change'})

env = factory.make_env(position_noise_m=0.0, yaw_noise_rad=0.0, render_cameras=True)
reset = D.make_reset(E, factory, env, SimpleNamespace(
    cpu=False, gripper_mode='pm1', max_steps=MAX_STEPS, replan=REPLAN, allow_invalid_scene=False))
store = {k: [] for k in ('eef', 'opening', 'cookies', 'src', 'rot')}
clip_log = []


def unwrapped():
    u = getattr(env, 'unwrapped', env)
    return u


def snap(obs, info):
    store['eef'].append(np.asarray(obs['observation.eef_pose'], np.float64))
    store['opening'].append(float(np.asarray(obs[STATE])[7]))
    store['cookies'].append(int(info.get('cookies_in_target', -1)))
    store['src'].append(int(info.get('cookies_in_source', -1)))
    try:
        u = unwrapped()
        ids = list(u._cookie_bodies)
        mats = np.stack([u.data.xmat[b].reshape(3, 3).copy() for b in ids])
        store['rot'].append(mats.astype(np.float32))
    except Exception:
        store['rot'].append(np.zeros((0, 3, 3), np.float32))


real_step = env.step


def step(action, **k):
    outv = real_step(action, **k)
    snap(outv[0], outv[4] if len(outv) > 4 else {})
    return outv


env.step = step


def reset_and_snapshot(env_, s):
    obs = reset(env_, s)
    snap(obs, {})
    return obs


real_infer_kwargs = policy._infer_kwargs


def clamped(obs, history, mask, ns):
    h = np.array(history, dtype=np.float64, copy=True)
    m = np.asarray(mask).astype(bool)
    if m.any():
        h[m] = np.clip(h[m], CLIP_LO, CLIP_HI)
    return real_infer_kwargs(obs, h.astype(np.asarray(history).dtype), mask, ns)


policy._infer_kwargs = clamped
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
t0 = time.time()
try:
    result = E.run_episode(env, policy, seed=SEED, reset=reset_and_snapshot,
                           max_steps=MAX_STEPS, replan=REPLAN)
finally:
    policy._infer_kwargs = real_infer_kwargs
    env.step = real_step
wall = round(time.time() - t0, 2)
rec, measured, chunks = D.serialize_result(result)

# final-state decomposition using the live env
u = unwrapped()
decomp = []
try:
    ids = list(u._cookie_bodies)
    tbid = u._target_bin_body
    tb_pos = u.data.xpos[tbid].copy()
    tb_mat = u.data.xmat[tbid].reshape(3, 3).copy()
    half_size = np.asarray(u.COOKIE_HALF_SIZE, np.float64)
    inner = np.asarray(u.TARGET_INNER_HALF_SIZE, np.float64)
    wall_h = float(u.TARGET_WALL_HEIGHT)
    tol = float(u.CONTACT_CONTAINMENT_TOLERANCE_M)
    floor_top = float(u.config.cookie_transfer.target_floor_z_m) + \
        float(u.config.cookie_transfer.bin_wall_thickness_m) / 2.0
    for i, bid in enumerate(ids):
        cpos = u.data.xpos[bid].copy()
        cmat = u.data.xmat[bid].reshape(3, 3).copy()
        p_rel = tb_mat.T @ (cpos - tb_pos)
        r_rel = tb_mat.T @ cmat
        half = np.abs(r_rel) @ half_size
        fp = bool(np.all(np.abs(p_rel[:2]) + half[:2] <= inner + tol))
        bottom = float(p_rel[2] - half[2])
        vert = bool(bottom >= floor_top - 0.003 and bottom < wall_h)
        rel = bool(bottom <= floor_top + 0.004)
        decomp.append({'cookie': i, 'world': cpos.round(5).tolist(),
                       'p_rel': p_rel.round(5).tolist(), 'half': half.round(5).tolist(),
                       'footprint_inside': fp, 'bottom': round(bottom, 5),
                       'vertically_inside': vert, 'released': rel,
                       'in_target_test': bool(fp and vert and rel),
                       'sim_mask': bool(u._cookie_inside_target(i))})
except Exception as exc:
    decomp = [{'error': repr(exc)}]

np.savez_compressed(OUT / ('probe_s%d.npz' % SEED),
                    eef=np.asarray(store['eef'], np.float64),
                    opening=np.asarray(store['opening'], np.float64),
                    cookies=np.asarray(store['cookies'], np.int64),
                    cookies_in_source=np.asarray(store['src'], np.int64),
                    rotations=np.asarray(store['rot'], np.float32),
                    measured=measured, **D.MARKERS)

n_fp = sum(1 for d in decomp if d.get('footprint_inside'))
n_vert = sum(1 for d in decomp if d.get('vertically_inside'))
n_rel = sum(1 for d in decomp if d.get('released'))
n_in = sum(1 for d in decomp if d.get('in_target_test'))
write('placement.json', {**D.MARKERS, 'seed': SEED, 'wall_s': wall,
                         'record': rec,
                         'constants': {'inner_half': inner.tolist(), 'wall_height': wall_h,
                                       'tol': tol, 'floor_top': floor_top,
                                       'cookie_half': half_size.tolist()},
                         'counts': {'footprint_inside': n_fp, 'vertically_inside': n_vert,
                                    'released': n_rel, 'in_target_test': n_in},
                         'in_target_cookies': [d for d in decomp if d.get('in_target_test')],
                         'near_misses': [d for d in decomp
                                         if d.get('footprint_inside') and not d.get('in_target_test')],
                         'all': decomp})
print('[probe s%d] wall=%.1fs fp=%d vert=%d rel=%d in=%d' % (SEED, wall, n_fp, n_vert, n_rel, n_in),
      flush=True)
print('ALL_DONE', flush=True)