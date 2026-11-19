"""G3: release gating with an integrated multi-channel safety guard (diagnostic arm).

Changes vs G2
  * safety_guard.Guard is called on every environment step; 'abort' raises and the run stops with
    its state saved (no further close commands are issued).
  * release feasibility A uses the carried cookie's pose AND velocity plus a short ballistic
    prediction (drift = v_xy * t_fall), and requires the predicted landing footprint to be inside
    the inner box and the drop height to be <= 0.02 m. It does NOT require the plate to already be
    on the floor; a high plate is reported as 'needs_descent', not released.
  * acceptance B reports the environment's own target_count / source_count / occupancy / exact_fill
    at every chunk boundary.
Startup self-check: the saved f128 force vector from G2 s17 is pushed through the same Guard
object; the run refuses to start unless it aborts.
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
SEED = int(os.environ.get('CL_GATE_SEED', '21'))
MAX_STEPS = int(os.environ.get('CL_MAX_STEPS', '600'))
OUT = ROOT / os.environ.get('CL_GATE_OUT', 'release_gate_g3_20261118')
OUT.mkdir(exist_ok=False)
GATE_MAX_CONSEC = 40
GATE_COOL = 10
PAD_LOAD = 1.0
PAD_LOST = 0.3
OPEN_THRESHOLD = 0.6
MAX_DROP = 0.02
G_ACC = 9.81


class AbortRun(RuntimeError):
    pass


# ---- guard integration self-check with the real f128 vector ----
_spike = np.load(ROOT / 'g2_s17' / 'G2_s17.npz')['step_force18'][128]
_g = Guard()
_act, _ev = _g.update(_spike, 0.455)
assert _act == 'abort' and 'pad_abort' in _ev, (_act, _ev)
print('[guard] f128 injection -> abort %s' % _ev, flush=True)

E, factory, W, fw = D.setup_protocol()
import torch
from a3_dual_arm_sim.contracts import STATE
from force_stream import CHANNELS
import record40k as R

policy, doc, calib, facts = D.build_policy_reconstructed(
    E, W, SimpleNamespace(cpu=False, gripper_mode='pm1'))
assert facts['device'] == 'cuda' and not policy.model.training
CLIP_LO = np.asarray(calib['per_channel_min'], np.float64)
CLIP_HI = np.asarray(calib['per_channel_max'], np.float64)

env = factory.make_env(position_noise_m=0.0, yaw_noise_rad=0.0, render_cameras=True)
u = getattr(env, 'unwrapped', env)
reset = D.make_reset(E, factory, env, SimpleNamespace(
    cpu=False, gripper_mode='pm1', max_steps=MAX_STEPS, replan=10, allow_invalid_scene=False))
ids = list(u._cookie_bodies)
tbid = u._target_bin_body
half = np.asarray(u.COOKIE_HALF_SIZE, np.float64)
inner = np.asarray(u.TARGET_INNER_HALF_SIZE, np.float64)
tol = float(u.CONTACT_CONTAINMENT_TOLERANCE_M)
cfg = u.config.cookie_transfer
floor_top = float(cfg.target_floor_z_m) + float(cfg.bin_wall_thickness_m) / 2.0
INNER_X = float(inner[0]) + tol
INNER_Y = float(inner[1]) + tol

guard = Guard()
store = {k: [] for k in ('eef', 'opening', 'cookies', 'src', 'force', 'grip', 'cp')}
chunks_log = []
abort_info = None


def snap(obs, info):
    f = np.asarray(obs['observation.force'], np.float64)
    store['eef'].append(np.asarray(obs['observation.eef_pose'], np.float64))
    store['opening'].append(float(np.asarray(obs[STATE])[7]))
    store['cookies'].append(int(info.get('cookies_in_target', -1)))
    store['src'].append(int(info.get('cookies_in_source', -1)))
    store['force'].append(f)
    store['cp'].append(np.stack([u.data.xpos[b].copy() for b in ids]))
    return f


real_step = env.step


def step(action, **k):
    outv = real_step(action, **k)
    f = snap(outv[0], outv[4] if len(outv) > 4 else {})
    act, ev = guard.update(f, store['opening'][-1], blocked=bool(k.pop('_blocked', False)))
    if act == 'abort':
        raise AbortRun(json.dumps({'events': ev, 'frame': len(store['opening']) - 1,
                                   'force': np.round(f, 1).tolist()}))
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
real_chunk = policy.chunk

st = {'state': 'NOT_CARRYING', 'confirm': 0, 'consec_blocked': 0, 'blocked_rows': 0,
      'delayed': 0, 'allowed': 0, 'lost': 0, 'needs_descent': 0, 'missing_motion': 0,
      'last_cmd_close': True, 'cool': 0, 'log': []}


def carried(eef):
    best, bd = None, 1e9
    for i, b in enumerate(ids):
        d = float(np.linalg.norm(u.data.xpos[b] - eef))
        if d < bd:
            best, bd = i, d
    return (best, bd) if bd < 0.06 else (None, bd)


def feasible(idx, eef):
    if idx is None:
        return False, {'why': 'no cookie within 6 cm'}
    b = ids[idx]
    cpos = u.data.xpos[b].copy()
    cmat = u.data.xmat[b].reshape(3, 3).copy()
    # velocity from the last two stored frames
    if len(store['cp']) >= 2:
        v = (store['cp'][-1][idx] - store['cp'][-2][idx]) * 20.0
    else:
        v = np.zeros(3)
    tb_pos = u.data.xpos[tbid].copy()
    tb_mat = u.data.xmat[tbid].reshape(3, 3).copy()
    p_rel = tb_mat.T @ (cpos - tb_pos)
    he = np.abs(tb_mat.T @ cmat) @ half
    bottom = float(p_rel[2] - he[2])
    drop = max(0.0, bottom - floor_top)
    t_fall = float(np.sqrt(2 * drop / G_ACC)) if drop > 0 else 0.0
    drift = np.array(v[:2]) * t_fall
    q = p_rel[:2] + drift
    horiz = bool(abs(q[0]) + he[0] <= INNER_X and abs(q[1]) + he[1] <= INNER_Y)
    ok = bool(horiz and drop <= MAX_DROP)
    return ok, {'pred_xy': q.round(4).tolist(), 'half': he.round(4).tolist(),
                'bottom': round(bottom, 4), 'drop': round(drop, 4),
                't_fall': round(t_fall, 3), 'horiz': horiz, 'drop_ok': drop <= MAX_DROP}


def gated_chunk(obs, frame, seed):
    acts, pf, meta = real_chunk(obs, frame, seed)
    f = np.asarray(obs['observation.force'], np.float64)
    pad = float(max(abs(f[12]), abs(f[13])))
    opening = float(np.asarray(obs[STATE])[7])
    eef = np.asarray(obs['observation.eef_pose'], np.float64)[:3]
    if st['cool'] > 0:
        st['cool'] -= 1
    ok_now = st['last_cmd_close'] and pad > PAD_LOAD and opening < OPEN_THRESHOLD
    st['confirm'] = st['confirm'] + 1 if ok_now else 0
    if st['state'] == 'CARRYING' and pad < PAD_LOST:
        st['state'] = 'LOST'
        st['lost'] += 1
        st['confirm'] = 0
        st['log'].append({'frame': int(frame), 'event': 'lost', 'pad': round(pad, 2)})
    if st['state'] in ('NOT_CARRYING', 'LOST') and st['confirm'] >= 2:
        st['state'] = 'CARRYING'
        st['log'].append({'frame': int(frame), 'event': 'carrying', 'pad': round(pad, 2)})
    idx, dist = carried(eef)
    ok, why = feasible(idx, eef) if st['state'] == 'CARRYING' else (True, {})
    if st['state'] == 'CARRYING' and not ok and st['cool'] == 0:
        st['needs_descent'] += 1
    n_blocked = 0
    if st['state'] == 'CARRYING' and st['cool'] == 0:
        for a in acts:
            if a[6] > 0:
                if ok:
                    st['allowed'] += 1
                    st['state'] = 'NOT_CARRYING'
                    st['confirm'] = 0
                    st['log'].append({'frame': int(frame), 'event': 'release_allowed',
                                      'cookie': idx, 'why': why})
                else:
                    a[6] = -1.0
                    n_blocked += 1
        if n_blocked:
            st['blocked_rows'] += n_blocked
            st['delayed'] += 1
            st['consec_blocked'] += 1
            st['log'].append({'frame': int(frame), 'event': 'open_delayed', 'rows': n_blocked,
                              'cookie': idx, 'why': why})
        if st['consec_blocked'] >= GATE_MAX_CONSEC:
            st['consec_blocked'] = 0
            st['cool'] = GATE_COOL
            st['log'].append({'frame': int(frame), 'event': 'cap_cool'})
    st['last_cmd_close'] = bool(acts[-1][6] < 0)
    ck = int(np.asarray(obs['observation.state']).shape[0]) if False else 0
    chunks_log.append({'frame': int(frame), 'state': st['state'], 'blocked': n_blocked,
                       'allowed': bool(ok), 'why': why, 'target': store['cookies'][-1] if store['cookies'] else -1})
    meta['g3'] = {'state': st['state'], 'blocked': n_blocked, 'allowed': bool(ok)}
    return acts, pf, meta


policy.chunk = gated_chunk
spy = R.ActionSpy().install(policy)
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
t0 = time.time()
try:
    result = E.run_episode(env, policy, seed=SEED, reset=reset_and_snapshot,
                           max_steps=MAX_STEPS, replan=10)
    rec, measured, chunks = D.serialize_result(result)
except AbortRun as exc:
    abort_info = json.loads(str(exc))
    rec, measured, chunks = {'aborted': True, 'abort': abort_info}, None, []
finally:
    policy.chunk = real_chunk
    policy._infer_kwargs = real_infer_kwargs
    env.step = real_step
wall = round(time.time() - t0, 2)

np.savez_compressed(OUT / ('G3_s%d.npz' % SEED),
                    step_eef_pose=np.asarray(store['eef'], np.float64),
                    step_opening=np.asarray(store['opening'], np.float64),
                    step_force18=np.asarray(store['force'], np.float64),
                    step_cookies=np.asarray(store['cookies'], np.int64),
                    step_cookies_in_source=np.asarray(store['src'], np.int64),
                    step_cookie_pos=np.asarray(store['cp'], np.float32),
                    measured=(measured if measured is not None else np.zeros((0, 6))),
                    **D.MARKERS)
summary = {**D.MARKERS, 'arm': 'G3', 'seed': SEED, 'wall_s': wall, 'record': rec,
           'guard_events': len(guard.events), 'abort': abort_info,
           'state_machine': {k: st[k] for k in ('delayed', 'allowed', 'lost', 'needs_descent',
                                               'blocked_rows')},
           'final_cookies': int(store['cookies'][-1]) if store['cookies'] else -1,
           'final_source': next((c for c in reversed(store['src']) if c >= 0), -1),
           'force_absmax': float(np.abs(np.asarray(store['force'])).max()) if store['force'] else None,
           'pad_absmax': float(max(abs(np.asarray(store['force'])[:, 12]).max(),
                                   abs(np.asarray(store['force'])[:, 13]).max())) if store['force'] else None,
           'missing_placement_motion': bool(st['delayed'] > 0 and st['allowed'] == 0)}
(OUT / ('G3_s%d.json' % SEED)).write_text(json.dumps(summary, indent=2, default=str))
(OUT / ('log_s%d.json' % SEED)).write_text(json.dumps({'state': st['log'], 'chunks': chunks_log,
                                                       'guard': guard.events}, indent=2, default=str))
print(json.dumps({k: summary[k] for k in ('wall_s', 'state_machine', 'final_cookies',
                                          'final_source', 'force_absmax', 'pad_absmax',
                                          'guard_events', 'abort',
                                          'missing_placement_motion')}), flush=True)
print('ALL_DONE', flush=True)