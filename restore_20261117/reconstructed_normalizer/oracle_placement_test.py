"""Oracle placement test: can the task's own success criterion ever return True?

Builds an ideal placement by writing the 10 target-slot positions straight into the cookie
free joints (bottom exactly on the bin floor, upright), calls the task's own predicates, and
then lets physics run to see whether the arrangement is stable. No policy, no training asset,
no modification of the criterion or the env.

Reported: the constants, per-cookie PASS/FAIL terms, slot occupancy, touches-all-walls,
exact_fill, and the same after N physics steps.
"""
import json
import os
import sys

import numpy as np

import diag_reconstructed_40k as D

OUT = '/workspace/eval_restore_20261117/reconstructed_normalizer/ORACLE_PLACEMENT.json'
N_STEPS = int(os.environ.get('ORACLE_STEPS', '60'))

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
tb_pos = data.xpos[tbid].copy()
tb_mat = data.xmat[tbid].reshape(3, 3).copy()
half = np.asarray(u.COOKIE_HALF_SIZE, np.float64)
inner = np.asarray(u.TARGET_INNER_HALF_SIZE, np.float64)
wall_h = float(u.TARGET_WALL_HEIGHT)
tol = float(u.CONTACT_CONTAINMENT_TOLERANCE_M)
cfg = u.config.cookie_transfer
floor_top = float(cfg.target_floor_z_m) + float(cfg.bin_wall_thickness_m) / 2.0
slots = np.asarray(cfg.target_slots_local_m, np.float64)

res = {
    'NORMALIZER_IDENTITY': 'RECONSTRUCTED_DETERMINISTIC_NOT_ORIGINAL',
    'original_stats_sha256': 'UNKNOWN',
    'scope': 'criterion validation only; cookie joints written directly, policy not involved',
    'constants': {'inner_half': inner.tolist(), 'tol': tol, 'floor_top': floor_top,
                  'wall_height': wall_h, 'cookie_half': half.tolist(),
                  'require_released': bool(getattr(u.task_config, 'require_released', None)),
                  'require_exact_slots': bool(getattr(u.task_config, 'require_exact_slots', None)),
                  'required_cookies': int(u.task_config.required_cookies),
                  'success_hold_steps': int(u.task_config.success_hold_steps),
                  'target_center_world': tb_pos.tolist()},
}

# ---- place the first 10 cookies into the 10 slots, bottom on the floor, upright ----
placed = []
for n, (sx, sy) in enumerate(slots):
    bid = ids[n]
    local = np.array([sx, sy, floor_top + half[2]])
    world = tb_pos + tb_mat @ local
    jadr = int(model.body_jntadr[bid])
    qadr = int(model.jnt_qposadr[jadr])
    data.qpos[qadr:qadr + 3] = world
    data.qpos[qadr + 3:qadr + 7] = [1.0, 0.0, 0.0, 0.0]
    data.qvel[int(model.jnt_dofadr[jadr]):int(model.jnt_dofadr[jadr]) + 6] = 0.0
    placed.append(int(n))
mujoco.mj_forward(model, data)


def decompose():
    out = []
    for i in range(len(ids)):
        cpos = data.xpos[ids[i]].copy()
        cmat = data.xmat[ids[i]].reshape(3, 3).copy()
        p_rel = tb_mat.T @ (cpos - tb_pos)
        r_rel = tb_mat.T @ cmat
        he = np.abs(r_rel) @ half
        fp = bool(np.all(np.abs(p_rel[:2]) + he[:2] <= inner + tol))
        bottom = float(p_rel[2] - he[2])
        vert = bool(bottom >= floor_top - 0.003 and bottom < wall_h)
        rel = bool(bottom <= floor_top + 0.004)
        out.append({'cookie': i, 'p_rel': p_rel.round(5).tolist(),
                    'half': he.round(5).tolist(), 'bottom': round(bottom, 5),
                    'footprint': fp, 'vertical': vert, 'released': rel,
                    'sim_in_target': bool(u._cookie_inside_target(i))})
    return out


def aggregate(dec):
    in_t = tuple(d['sim_in_target'] for d in dec)
    occ = list(u._target_slot_occupancy(in_t))
    touches = bool(u._collection_touches_all_walls(in_t, u.TARGET_CENTER,
                                                   u.TARGET_INNER_HALF_SIZE, is_target=True))
    target_count = int(sum(in_t))
    src = tuple(u._cookie_inside_source(i) for i in range(len(ids)))
    source_count = int(sum(src))
    exact = bool(target_count == u.task_config.required_cookies
                 and source_count == len(ids) - u.task_config.required_cookies
                 and all(o >= 0 for o in occ) and touches)
    count_fill = bool(target_count == u.task_config.required_cookies
                      and source_count == len(ids) - u.task_config.required_cookies)
    return {'target_count': target_count, 'source_count': source_count,
            'occupancy': occ, 'touches_all_walls': touches,
            'count_fill': count_fill, 'exact_fill': exact,
            'criterion_used': 'exact_slots' if u.task_config.require_exact_slots else 'released_count'}


dec0 = decompose()
print('IMMEDIATE', json.dumps(aggregate(dec0)), flush=True)
res['immediately_after_write'] = {'placed': placed,
                                  'aggregate': aggregate(dec0),
                                  'cookies': [d for d in dec0[:12]]}

# ---- let physics run with zero control ----
data.ctrl[:] = 0.0
for _ in range(N_STEPS):
    mujoco.mj_step(model, data)
dec1 = decompose()
print('AFTER_PHYSICS', json.dumps(aggregate(dec1)), flush=True)
res['after_physics_steps'] = {'steps': N_STEPS, 'aggregate': aggregate(dec1),
                              'cookies': [d for d in dec1[:12]]}

# ---- also try the task's own step() loop to exercise the success hold counter ----
try:
    u._success_hold_count = 0
    a = np.zeros(14, dtype=np.float64)
    ok_steps = 0
    best = 0
    for _ in range(25):
        _obs, _rew, _term, _trunc, info = u.step(a)
        best = max(best, int(info.get('success_hold_count', 0)))
        ok_steps += int(bool(info.get('success')))
    res['task_step_loop'] = {'success_any': bool(ok_steps), 'max_hold_count': best,
                             'final_info': {k: info.get(k) for k in
                                            ('success', 'cookies_in_target', 'cookies_in_source',
                                             'target_touches_all_walls', 'target_slot_occupancy',
                                             'success_hold_count')}}
except Exception as exc:
    res['task_step_loop'] = {'error': repr(exc)}

print('STEP_LOOP', json.dumps(res.get('task_step_loop', {})), flush=True)
with open(OUT, 'w') as fh:
    json.dump(res, fh, indent=2, default=str)
print(json.dumps(res, indent=1, default=str)[:5000])
print('WROTE', OUT)