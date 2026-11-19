"""Teacher-data collector (independent branch, CANDIDATE_TEACHER_DATA).

Renders the simulator while the P2-clamped + G3-guided policy runs, and stores per step:
  front / wrist images (as rendered), 14D state, force18, opening, executed grip command,
  the teacher decision for the chunk that produced that step (hold / allow release + reason),
  and the environment's own success counters.

The teacher decisions are training targets only; they are never fed back into the student input.
Bounded: TEACHER_SEEDS x 600 steps, no training, no asset modification.
"""
import json
import os
import random
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import diag_reconstructed_40k as D
from safety_guard import Guard

ROOT = Path(__file__).resolve().parent
OUT = ROOT / os.environ.get('TEACHER_OUT', 'teacher_data_20261118')
OUT.mkdir(exist_ok=False)
SEEDS = [int(s) for s in os.environ.get('TEACHER_SEEDS', '21,24').split(',') if s.strip()]
MAX_STEPS = int(os.environ.get('CL_MAX_STEPS', '600'))
GATE_MAX_CONSEC = 40
GATE_COOL = 10
PAD_LOAD, PAD_LOST, OPEN_TH = 1.0, 0.3, 0.6
MAX_DROP = 0.02

E, factory, W, fw = D.setup_protocol()
import torch
from a3_dual_arm_sim.contracts import STATE
from force_stream import CHANNELS

policy, doc, calib, facts = D.build_policy_reconstructed(
    E, W, SimpleNamespace(cpu=False, gripper_mode='pm1'))
assert facts['device'] == 'cuda'
CLIP_LO = np.asarray(calib['per_channel_min'], np.float64)
CLIP_HI = np.asarray(calib['per_channel_max'], np.float64)

env = factory.make_env(position_noise_m=0.0, yaw_noise_rad=0.0, render_cameras=True)
u = getattr(env, 'unwrapped', env)
ids = list(u._cookie_bodies)
tbid = u._target_bin_body
half = np.asarray(u.COOKIE_HALF_SIZE, np.float64)
inner = np.asarray(u.TARGET_INNER_HALF_SIZE, np.float64)
tol = float(u.CONTACT_CONTAINMENT_TOLERANCE_M)
cfg = u.config.cookie_transfer
floor_top = float(cfg.target_floor_z_m) + float(cfg.bin_wall_thickness_m) / 2.0

IMG_KEYS = None
guard = Guard()
teach = {'state': 'NOT_CARRYING', 'confirm': 0, 'cool': 0, 'last_close': True}


def carried(eef):
    best, bd = None, 1e9
    for i, b in enumerate(ids):
        d = float(np.linalg.norm(u.data.xpos[b] - eef))
        if d < bd:
            best, bd = i, d
    return (best, bd) if bd < 0.06 else (None, bd)


def feasible(idx):
    if idx is None:
        return False, 'no_cookie'
    b = ids[idx]
    cpos = u.data.xpos[b].copy()
    cmat = u.data.xmat[b].reshape(3, 3).copy()
    tb_pos = u.data.xpos[tbid].copy()
    tb_mat = u.data.xmat[tbid].reshape(3, 3).copy()
    p_rel = tb_mat.T @ (cpos - tb_pos)
    he = np.abs(tb_mat.T @ cmat) @ half
    bottom = float(p_rel[2] - he[2])
    drop = max(0.0, bottom - floor_top)
    horiz = bool(abs(p_rel[0]) + he[0] <= inner[0] + tol and abs(p_rel[1]) + he[1] <= inner[1] + tol)
    ok = bool(horiz and drop <= MAX_DROP)
    why = 'ok' if ok else ('high' if not (drop <= MAX_DROP) else 'off_box')
    return ok, why


def run(seed):
    global IMG_KEYS
    reset = D.make_reset(E, factory, env, SimpleNamespace(
        cpu=False, gripper_mode='pm1', max_steps=MAX_STEPS, replan=10, allow_invalid_scene=False))
    store = {k: [] for k in ('img0', 'img1', 'state', 'force', 'opening', 'grip', 'teach', 'tc', 'sc')}
    tdec = []

    real_step = env.step

    def step(action, **k):
        outv = real_step(action, **k)
        obs, info = outv[0], (outv[4] if len(outv) > 4 else {})
        global IMG_KEYS
        if IMG_KEYS is None:
            IMG_KEYS = sorted([k2 for k2 in obs if k2.startswith('observation.images')])
            print('[imgs]', IMG_KEYS, flush=True)
        store['img0'].append(np.asarray(obs[IMG_KEYS[0]], np.uint8).copy())
        store['img1'].append(np.asarray(obs[IMG_KEYS[1]], np.uint8).copy())
        store['state'].append(np.asarray(obs[STATE], np.float64))
        f = np.asarray(obs['observation.force'], np.float64)
        store['force'].append(f)
        store['opening'].append(float(np.asarray(obs[STATE])[7]))
        store['teach'].append(tdec[-1] if tdec else -2)
        store['tc'].append(int(info.get('cookies_in_target', -1)))
        store['sc'].append(int(info.get('cookies_in_source', -1)))
        act, ev = guard.update(f, store['opening'][-1])
        if act == 'abort':
            raise RuntimeError('SAFETY_ABORT ' + json.dumps({'ev': ev, 'frame': len(store['opening']) - 1}))
        return outv

    env.step = step

    def reset_snap(env_, s):
        obs = reset(env_, s)
        global IMG_KEYS
        if IMG_KEYS is None:
            IMG_KEYS = sorted([k2 for k2 in obs if k2.startswith('observation.images')])
            print('[imgs]', IMG_KEYS, flush=True)
        store['img0'].append(np.asarray(obs[IMG_KEYS[0]], np.uint8).copy())
        store['img1'].append(np.asarray(obs[IMG_KEYS[1]], np.uint8).copy())
        store['state'].append(np.asarray(obs[STATE], np.float64))
        store['force'].append(np.asarray(obs['observation.force'], np.float64))
        store['opening'].append(float(np.asarray(obs[STATE])[7]))
        store['teach'].append(-1)
        store['tc'].append(0)
        store['sc'].append(80)
        return obs

    real_kw = policy._infer_kwargs

    def clamped(obs, history, mask, ns):
        h = np.array(history, dtype=np.float64, copy=True)
        m = np.asarray(mask).astype(bool)
        if m.any():
            h[m] = np.clip(h[m], CLIP_LO, CLIP_HI)
        return real_kw(obs, h.astype(np.asarray(history).dtype), mask, ns)

    policy._infer_kwargs = clamped
    real_chunk = policy.chunk

    def gated(obs, frame, seed):
        acts, pf, meta = real_chunk(obs, frame, seed)
        f = np.asarray(obs['observation.force'], np.float64)
        pad = float(max(abs(f[12]), abs(f[13])))
        eef = np.asarray(obs['observation.eef_pose'], np.float64)[:3]
        opening = float(np.asarray(obs[STATE])[7])
        ok_now = teach['last_close'] and pad > PAD_LOAD and opening < OPEN_TH
        teach['confirm'] = teach['confirm'] + 1 if ok_now else 0
        if teach['state'] == 'CARRYING' and pad < PAD_LOST:
            teach['state'] = 'NOT_CARRYING'
            teach['confirm'] = 0
        if teach['state'] != 'CARRYING' and teach['confirm'] >= 2:
            teach['state'] = 'CARRYING'
        idx, _ = carried(eef)
        ok, why = feasible(idx)
        label = 0
        if teach['state'] == 'CARRYING':
            for a in acts:
                if a[6] > 0:
                    if ok:
                        label = 1                       # teacher: release allowed here
                        teach['state'] = 'NOT_CARRYING'
                        teach['confirm'] = 0
                    else:
                        a[6] = -1.0                     # teacher: keep closing (high / off box)
                        label = -1 if why == 'high' else -2
            tdec.append(label)
        else:
            tdec.append(-2)
        teach['last_close'] = bool(acts[-1][6] < 0)
        return acts, pf, meta

    policy.chunk = gated
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    t0 = time.time()
    rec = None
    try:
        result = E.run_episode(env, policy, seed=seed, reset=reset_snap,
                               max_steps=MAX_STEPS, replan=10)
        rec, measured, _ = D.serialize_result(result)
    except RuntimeError as exc:
        rec = {'aborted': str(exc)}
    finally:
        policy.chunk = real_chunk
        policy._infer_kwargs = real_kw
        env.step = real_step
    wall = round(time.time() - t0, 2)

    np.savez(OUT / ('teacher_s%d.npz' % seed),
             img_front=np.asarray(store['img0'], np.uint8),
             img_wrist=np.asarray(store['img1'], np.uint8),
             state=np.asarray(store['state'], np.float64),
             force18=np.asarray(store['force'], np.float64),
             opening=np.asarray(store['opening'], np.float64),
             teach=np.asarray(store['teach'], np.int64),
             cookies_in_target=np.asarray(store['tc'], np.int64),
             cookies_in_source=np.asarray(store['sc'], np.int64),
             **D.MARKERS)
    summ = {**D.MARKERS, 'seed': seed, 'steps': len(store['opening']), 'wall_s': wall,
            'record': rec, 'guard_events': len(guard.events),
            'image_keys': IMG_KEYS,
            'teach_counts': {str(k): int((np.asarray(store['teach']) == k).sum())
                             for k in (-2, -1, 0, 1)},
            'final_cookies': store['tc'][-1], 'final_source': store['sc'][-1],
            'bytes': int(sum(a.nbytes for a in (store['img0'], store['img1'])))}
    (OUT / ('teacher_s%d.json' % seed)).write_text(json.dumps(summ, indent=2, default=str))
    print('[seed %d] %s' % (seed, json.dumps({k: summ[k] for k in
                                              ('steps', 'wall_s', 'guard_events', 'teach_counts',
                                               'final_cookies', 'final_source')})), flush=True)


for s in SEEDS:
    try:
        run(s)
    except Exception as exc:
        (OUT / ('teacher_s%d.json' % s)).write_text(json.dumps({'seed': s, 'error': repr(exc)}, indent=2))
        print('[seed %d] failed: %r' % (s, exc), flush=True)
print('ALL_DONE', flush=True)