"""Clean placement counterfactual (A/B/C/D) from the real failing final state. CPU only.

A  original final state, untouched
B  arm only: all non-cookie, non-bin joints restored to their reset values (arm parked high)
C  cookies only: ten distinct slots assigned by global linear-sum matching, each selected
   cookie placed upright with its bottom on the bin floor; arm left where it was
D  both B and C

Each variant is evaluated at steps 0, 1, 20 and 60 (mj_step with zero control). Reported per
sample: target_count, exact_fill, occupancy, number of contacts involving cookie bodies, the
largest contact force on a cookie, and the most negative contact distance (penetration).

This moves simulator bodies directly. It is a diagnostic oracle, not policy execution.
"""
import json
import os
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment

import diag_reconstructed_40k as D

OUT = '/workspace/eval_restore_20261117/reconstructed_normalizer/PLACEMENT_ORACLE_ABCD.json'
FINAL = '/workspace/eval_restore_20261117/reconstructed_normalizer/placement_counterfactual_20261118_v2/final_state.npz'
SAMPLES = [0, 1, 20, 60]

E, factory, W, fw = D.setup_protocol()
import mujoco

env = factory.make_env(position_noise_m=0.0, yaw_noise_rad=0.0, render_cameras=True)
u = getattr(env, 'unwrapped', env)
reset = D.make_reset(E, factory, env, type('A', (), {
    'cpu': False, 'gripper_mode': 'pm1', 'max_steps': 600, 'replan': 10,
    'allow_invalid_scene': False})())
obs = reset(env, 17)

model, data = u.model, u.data
ids = list(u._cookie_bodies)
tbid = u._target_bin_body
qpos_reset = data.qpos.copy()
qvel_reset = data.qvel.copy()

tb_pos0 = data.xpos[tbid].copy()
tb_mat = data.xmat[tbid].reshape(3, 3).copy()
half = np.asarray(u.COOKIE_HALF_SIZE, np.float64)
inner = np.asarray(u.TARGET_INNER_HALF_SIZE, np.float64)
tol = float(u.CONTACT_CONTAINMENT_TOLERANCE_M)
cfg = u.config.cookie_transfer
floor_top = float(cfg.target_floor_z_m) + float(cfg.bin_wall_thickness_m) / 2.0
slots = np.asarray(cfg.target_slots_local_m, np.float64)
cookie_body_ids = set(int(b) for b in ids)


def joint_range(bid):
    j = int(model.body_jntadr[bid])
    return int(model.jnt_qposadr[j]), int(model.jnt_dofadr[j])


range_cookie = [joint_range(b) for b in ids]
range_bin = joint_range(tbid)


def arm_mask():
    m = np.ones(model.nq, dtype=bool)
    for qa, _ in range_cookie:
        m[qa:qa + 7] = False
    m[range_bin[0]:range_bin[0] + 7] = False
    return m


ARM = arm_mask()


def arm_dof_mask():
    m = np.ones(model.nv, dtype=bool)
    for _, va in range_cookie:
        m[va:va + 6] = False
    m[range_bin[1]:range_bin[1] + 6] = False
    return m


ARM_DOF = arm_dof_mask()


def contact_stats():
    n = 0
    fmax = 0.0
    dmin = 0.0
    for i in range(data.ncon):
        c = data.contact[i]
        b1 = int(model.geom_bodyid[c.geom1])
        b2 = int(model.geom_bodyid[c.geom2])
        f = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, f)
        mag = float(np.linalg.norm(f[:3]))
        if b1 in cookie_body_ids or b2 in cookie_body_ids:
            n += 1
            fmax = max(fmax, mag)
            dmin = min(dmin, float(c.dist))
    return {'cookie_contacts': n, 'max_cookie_contact_force': round(fmax, 2),
            'min_contact_dist': round(dmin, 5)}


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
    return {'target_count': tc, 'source_count': sc, 'exact_fill': exact,
            'occupancy': occ, 'in_target_ids': [i for i, v in enumerate(in_t) if v],
            'contacts': contact_stats()}


z = np.load(FINAL)
qpos0 = z['qpos'].copy()
qvel0 = z['qvel'].copy()
cp0 = z['cookie_pos'].astype(np.float64)
SCENE = '/workspace/eval_restore_20261117/reconstructed_normalizer/closed_loop_scene_20261118_v2/CL_s17.npz'
first = np.asarray(np.load(SCENE)['step_cookie_pos'], np.float64)[0]
moved = np.flatnonzero(np.linalg.norm(cp0 - first, axis=1) > 0.020).tolist()
print('[moved]', moved, flush=True)


def restore():
    data.qpos[:] = qpos0
    data.qvel[:] = qvel0
    mujoco.mj_forward(model, data)


def assign_slots():
    """Global unique assignment between moved cookies and slots (linear_sum_assignment)."""
    cost = np.zeros((len(moved), len(slots)))
    for i, c in enumerate(moved):
        rel = tb_mat.T @ (data.xpos[ids[c]] - tb_pos0)
        for j, sxy in enumerate(slots):
            cost[i, j] = np.hypot(rel[0] - sxy[0], rel[1] - sxy[1])
    ri, ci = linear_sum_assignment(cost)
    pairs = sorted(zip(cost[ri, ci], ri, ci))[:len(slots)]
    return [(int(moved[r]), int(c)) for _, r, c in pairs], cost


def variant(kind):
    restore()
    info = {'arm_retracted': False, 'cookies_corrected': 0}
    if kind in ('B', 'D'):
        data.qpos[ARM] = qpos_reset[ARM]
        data.qvel[ARM_DOF] = 0.0
        mujoco.mj_forward(model, data)
        info['arm_retracted'] = True
    if kind == 'E':
        restore()
        data.qpos[ARM] = qpos_reset[ARM]
        data.qvel[ARM_DOF] = 0.0
        mujoco.mj_forward(model, data)
        info['arm_retracted'] = True
    if kind in ('C', 'D', 'E'):
        pairs, _ = assign_slots()
        for c, si in pairs:
            qa, va = range_cookie[c]
            world = tb_pos0 + tb_mat @ np.array([slots[si][0], slots[si][1],
                                                 floor_top + half[2] + 0.001])
            data.qpos[qa:qa + 3] = world
            data.qpos[qa + 3:qa + 7] = [1.0, 0.0, 0.0, 0.0]
            data.qvel[va:va + 6] = 0.0
        info['cookies_corrected'] = len(pairs)
        info['assignment'] = {int(c): int(s) for c, s in pairs}
        mujoco.mj_forward(model, data)
        if kind == 'E':
            in_t = tuple(bool(u._cookie_inside_target(i)) for i in range(len(ids)))
            in_s = tuple(bool(u._cookie_inside_source(i)) for i in range(len(ids)))
            stray = [i for i in range(len(ids)) if not in_t[i] and not in_s[i]]
            for i in stray:
                qa, va = range_cookie[i]
                data.qpos[qa:qa + 3] = first[i]
                data.qpos[qa + 3:qa + 7] = [1.0, 0.0, 0.0, 0.0]
                data.qvel[va:va + 6] = 0.0
            info['stray_returned'] = stray
        mujoco.mj_forward(model, data)
    return info


res = {**D.MARKERS, 'scope': 'oracle counterfactual on the real failing final state; moves bodies '
                            'directly, NOT policy execution',
       'seed': 17, 'moved_cookies': moved, 'samples': SAMPLES,
       'constants': {'inner_half': inner.tolist(), 'tol': tol, 'floor_top': floor_top,
                     'cookie_half': half.tolist(), 'slots': slots.tolist()},
       'variants': {}}

for kind in ('A', 'B', 'C', 'D', 'E'):
    info = variant(kind)
    trace = {}
    data.ctrl[:] = 0.0
    for n in range(max(SAMPLES) + 1):
        if n in SAMPLES:
            trace[str(n)] = evaluate()
        mujoco.mj_step(model, data)
    res['variants'][kind] = {**info, 'trace': trace}
    print('[%s] corrected=%d retracted=%s | t0 tc=%d ex=%s | t20 tc=%d ex=%s | t60 tc=%d ex=%s'
          % (kind, info['cookies_corrected'], info['arm_retracted'],
             trace['0']['target_count'], trace['0']['exact_fill'],
             trace['20']['target_count'], trace['20']['exact_fill'],
             trace['60']['target_count'], trace['60']['exact_fill']), flush=True)

with open(OUT, 'w') as fh:
    json.dump(res, fh, indent=2, default=str)
print('WROTE', OUT)