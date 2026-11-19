"""Geometric counterfactuals on a real failing final state (single diagnostic run).

Runs one P2-clamped episode (seed 17), snapshots the final simulator state, and then replays
four single-factor corrections on copies of that state:

  C   no correction, only extra settle (60 physics steps)
  A   horizontal only: each moved cookie's xy is moved to its nearest target slot, z untouched
  B   vertical only:   each moved cookie's bottom is set exactly on the bin floor, xy untouched
  D   both A and B

After each correction the task's own predicates decide target_count / occupancy /
touches_all_walls / exact_fill, before and after the settle. This measures how much of the
failure is horizontal placement, how much is release height, and whether either alone is enough.

Nothing is executed with a modified policy; the correction is applied to a copy of the final
diagnostic state only.
"""
import copy
import json
import os
import random
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import diag_reconstructed_40k as D

ROOT = Path(__file__).resolve().parent
SEED = int(os.environ.get('CL_CF_SEED', '17'))
MAX_STEPS = int(os.environ.get('CL_MAX_STEPS', '600'))
REPLAN = 10
SETTLE = int(os.environ.get('CL_CF_SETTLE', '60'))
OUT = ROOT / os.environ.get('CL_CF_OUT', 'placement_counterfactual_20261118')
OUT.mkdir(exist_ok=False)

E, factory, W, fw = D.setup_protocol()
import torch
import mujoco
from a3_dual_arm_sim.contracts import STATE
from force_stream import CHANNELS, CHANNEL_NAMES

policy, doc, calib, facts = D.build_policy_reconstructed(
    E, W, SimpleNamespace(cpu=False, gripper_mode='pm1'))
assert facts['device'] == 'cuda' and not policy.model.training
CLIP_LO = np.asarray(calib['per_channel_min'], dtype=np.float64)
CLIP_HI = np.asarray(calib['per_channel_max'], dtype=np.float64)

env = factory.make_env(position_noise_m=0.0, yaw_noise_rad=0.0, render_cameras=True)
u = getattr(env, 'unwrapped', env)
reset = D.make_reset(E, factory, env, SimpleNamespace(
    cpu=False, gripper_mode='pm1', max_steps=MAX_STEPS, replan=REPLAN, allow_invalid_scene=False))

store = {'cp': [], 'eef': [], 'grip': []}
real_step = env.step


def snap(obs, info):
    store['eef'].append(np.asarray(obs['observation.eef_pose'], np.float64)[:3])
    try:
        uu = getattr(env, 'unwrapped', env)
        store['cp'].append(np.stack([uu.data.xpos[b].copy() for b in uu._cookie_bodies]))
    except Exception:
        store['cp'].append(np.zeros((0, 3)))
    store['grip'].append(info.get('cookies_in_target', -1))


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

model, data = u.model, u.data
ids = list(u._cookie_bodies)
tbid = u._target_bin_body
tb_pos = data.xpos[tbid].copy()
tb_mat = data.xmat[tbid].reshape(3, 3).copy()
half = np.asarray(u.COOKIE_HALF_SIZE, np.float64)
inner = np.asarray(u.TARGET_INNER_HALF_SIZE, np.float64)
tol = float(u.CONTACT_CONTAINMENT_TOLERANCE_M)
cfg = u.config.cookie_transfer
floor_top = float(cfg.target_floor_z_m) + float(cfg.bin_wall_thickness_m) / 2.0
slots = np.asarray(cfg.target_slots_local_m, np.float64)

qpos0 = data.qpos.copy()
qvel0 = data.qvel.copy()
cp0 = np.stack([data.xpos[b].copy() for b in ids])
moved = np.flatnonzero(np.linalg.norm(cp0 - np.asarray(store['cp'][0]), axis=1) > 0.020)


def qadr_of(bid):
    j = int(model.body_jntadr[bid])
    return int(model.jnt_qposadr[j]), int(model.jnt_dofadr[j])


def evaluate():
    in_t = tuple(bool(u._cookie_inside_target(i)) for i in range(len(ids)))
    occ = list(u._target_slot_occupancy(in_t))
    touches = bool(u._collection_touches_all_walls(in_t, u.TARGET_CENTER,
                                                   u.TARGET_INNER_HALF_SIZE, is_target=True))
    tc = int(sum(in_t))
    sc = int(sum(bool(u._cookie_inside_source(i)) for i in range(len(ids))))
    exact = bool(tc == u.task_config.required_cookies
                 and sc == len(ids) - u.task_config.required_cookies
                 and all(o >= 0 for o in occ) and touches)
    return {'target_count': tc, 'source_count': sc, 'occupancy': occ,
            'touches_all_walls': touches, 'exact_fill': exact,
            'in_target_ids': [i for i, v in enumerate(in_t) if v]}


def restore(qp, qv):
    data.qpos[:] = qp
    data.qvel[:] = qv
    mujoco.mj_forward(model, data)


def assign_slots():
    """Ten distinct slots for the ten moved cookies that are closest to any slot."""
    cand = []
    for c in moved:
        rel = tb_mat.T @ (data.xpos[ids[c]] - tb_pos)
        d = [float(np.hypot(rel[0] - sxy[0], rel[1] - sxy[1])) for sxy in slots]
        cand.append((min(d), c, int(np.argmin(d))))
    cand.sort()
    used, assign = set(), {}
    for _, c, si in cand:
        if si in used:
            continue
        used.add(si)
        assign[int(c)] = si
        if len(assign) == len(slots):
            break
    return assign


def restore_checked():
    restore(qpos0, qvel0)
    d = float(np.abs(np.stack([data.xpos[b] for b in ids]) - cp0).max())
    return d


def apply_variant(kind, assign):
    drift = restore_checked()
    n = 0
    for c, si in assign.items():
        bid = ids[c]
        qa, va = qadr_of(bid)
        p = data.qpos[qa:qa + 3].copy()
        rel = tb_mat.T @ (p - tb_pos)
        if kind in ('A', 'D'):
            rel = np.array([slots[si][0], slots[si][1], rel[2]])
        if kind in ('B', 'D'):
            rel[2] = floor_top + half[2] + 0.001   # bottom 1 mm above the floor
        world = tb_pos + tb_mat @ rel
        data.qpos[qa:qa + 3] = world
        data.qpos[qa + 3:qa + 7] = [1.0, 0.0, 0.0, 0.0]
        data.qvel[va:va + 6] = 0.0
        n += 1
    mujoco.mj_forward(model, data)
    return n, drift


res = {**D.MARKERS, 'scope': 'diagnostic counterfactual on a real failing final state',
       'seed': SEED, 'max_steps': MAX_STEPS, 'settle_steps': SETTLE,
       'constants': {'inner_half': inner.tolist(), 'tol': tol, 'floor_top': floor_top,
                     'cookie_half': half.tolist()},
       'record': rec, 'wall_s': wall,
       'moved_cookies': moved.tolist(), 'variants': {}}

restore(qpos0, qvel0)
res['variants']['baseline'] = {'before_settle': evaluate()}
data.ctrl[:] = 0.0
for _ in range(SETTLE):
    mujoco.mj_step(model, data)
res['variants']['baseline']['after_settle'] = evaluate()

ASSIGN = assign_slots()
res['slot_assignment'] = {str(k): int(v) for k, v in ASSIGN.items()}
print('[assign]', res['slot_assignment'], flush=True)
for kind in ('C', 'A', 'B', 'D'):
    n, drift = apply_variant(kind, ASSIGN)
    before = evaluate()
    data.ctrl[:] = 0.0
    for _ in range(SETTLE):
        mujoco.mj_step(model, data)
    after = evaluate()
    res['variants'][kind] = {'corrected_cookies': n, 'restore_drift_m': drift,
                             'before_settle': before, 'after_settle': after}
    print('[%s] corrected=%d before tc=%d exact=%s | after tc=%d exact=%s'
          % (kind, n, before['target_count'], before['exact_fill'],
             after['target_count'], after['exact_fill']), flush=True)

with open(OUT / 'counterfactual.json', 'w') as fh:
    json.dump(res, fh, indent=2, default=str)
np.savez_compressed(OUT / 'final_state.npz', qpos=qpos0, qvel=qvel0,
                    cookie_pos=cp0.astype(np.float32), **D.MARKERS)
print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != 'in_target_ids'}
                  for k, v in res['variants'].items()}, indent=1), flush=True)
print('ALL_DONE', flush=True)