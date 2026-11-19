"""G2: release gating driven by a carrying state machine and the carried cookie's pose.

State machine (updated only from observations at chunk boundaries, never from the pending command)
  NOT_CARRYING -> CARRYING   two consecutive boundaries with: previous chunk ended closed,
                             pad force > 1 N, measured opening < 0.6
  CARRYING     -> LOST       pad force < 0.3 N at a boundary (plate gone)
  CARRYING     -> NOT_CARRYING after an allowed release
  any          -> SAFETY_ABORT (suspend gating 10 steps) when wrist force > 150 N

Only in CARRYING does a new open request get evaluated. The release is allowed when the carried
cookie's own geometry passes:
  A  over the box: |cookie.xy - target.xy| + footprint half extent <= inner half + 0.02
  B  low enough:  cookie bottom (bin-local z) <= 0.06 (rim is 0.026)
Otherwise the open request is delayed (close is commanded instead) and counted. 40-step cap and
150 N suspension as before.

The cookie pose is simulator truth: this is a diagnostic arm, not a deployable controller.
"""
import json
import os
import random
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import diag_reconstructed_40k as D

ROOT = Path(__file__).resolve().parent
SEED = int(os.environ.get('CL_GATE_SEED', '21'))
MAX_STEPS = int(os.environ.get('CL_MAX_STEPS', '600'))
OUT = ROOT / os.environ.get('CL_GATE_OUT', 'release_gate_g2_20261118')
OUT.mkdir(exist_ok=False)
GATE_MAX_CONSEC = 40
GATE_COOL = 10
FORCE_SUSPEND = 150.0
PAD_LOAD = 1.0
PAD_LOST = 0.3
OPEN_THRESHOLD = 0.6

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

store = {k: [] for k in ('eef', 'opening', 'cookies', 'src', 'force', 'pad')}
transitions = []


def snap(obs, info):
    store['eef'].append(np.asarray(obs['observation.eef_pose'], np.float64))
    store['opening'].append(float(np.asarray(obs[STATE])[7]))
    store['cookies'].append(int(info.get('cookies_in_target', -1)))
    store['src'].append(int(info.get('cookies_in_source', -1)))
    f = np.asarray(obs['observation.force'], np.float64)
    store['force'].append(f)
    store['pad'].append(float(max(abs(f[12]), abs(f[13]))))


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
real_chunk = policy.chunk

st = {'state': 'NOT_CARRYING', 'confirm': 0, 'consec_blocked': 0, 'cool': 0,
      'blocked_rows': 0, 'blocked_events': 0, 'releases': 0, 'lost': 0, 'aborts': 0,
      'last_cmd_close': True, 'log': []}


def carried_cookie(eef):
    """Cookie nearest to the EEF within 6 cm (simulator truth)."""
    best, bd = None, 1e9
    for i, b in enumerate(ids):
        p = u.data.xpos[b]
        d = float(np.linalg.norm(p - eef))
        if d < bd:
            best, bd = i, d
    return (best, bd) if bd < 0.06 else (None, bd)


def release_ok(idx, eef):
    if idx is None:
        return False, {'why': 'no cookie within 6 cm'}
    b = ids[idx]
    cpos = u.data.xpos[b].copy()
    cmat = u.data.xmat[b].reshape(3, 3).copy()
    tb_pos = u.data.xpos[tbid].copy()
    tb_mat = u.data.xmat[tbid].reshape(3, 3).copy()
    p_rel = tb_mat.T @ (cpos - tb_pos)
    he = np.abs(tb_mat.T @ cmat) @ half
    over = bool(np.all(np.abs(p_rel[:2]) + he[:2] <= inner + 0.02))
    bottom = float(p_rel[2] - he[2])
    low = bool(bottom <= 0.06)
    return bool(over and low), {'over_box': over, 'low': low,
                                'bottom': round(bottom, 4),
                                'p_rel': p_rel.round(4).tolist()}


def gated_chunk(obs, frame, seed):
    acts, pf, meta = real_chunk(obs, frame, seed)
    f = np.asarray(obs['observation.force'], np.float64)
    pad = float(max(abs(f[12]), abs(f[13])))
    opening = float(np.asarray(obs[STATE])[7])
    wrist = float(np.linalg.norm(f[[int(c) for c in CHANNELS][:3]]))
    eef = np.asarray(obs['observation.eef_pose'], np.float64)[:3]

    # ---- state update from the PREVIOUS chunk's facts ----
    if st['cool'] > 0:
        st['cool'] -= 1
    elif wrist > FORCE_SUSPEND:
        st['aborts'] += 1
        st['cool'] = 10
        st['consec_blocked'] = 0
        st['log'].append({'frame': int(frame), 'state': st['state'], 'event': 'force_suspend',
                          'wrist': round(wrist, 1)})
    ok_now = st['last_cmd_close'] and pad > PAD_LOAD and opening < OPEN_THRESHOLD
    st['confirm'] = st['confirm'] + 1 if ok_now else 0
    if st['state'] == 'CARRYING' and pad < PAD_LOST:
        st['state'] = 'LOST'
        st['lost'] += 1
        st['log'].append({'frame': int(frame), 'state': 'LOST', 'event': 'pad_lost',
                          'pad': round(pad, 2)})
        st['confirm'] = 0
    if st['state'] in ('NOT_CARRYING', 'LOST') and st['confirm'] >= 2:
        st['state'] = 'CARRYING'
        st['log'].append({'frame': int(frame), 'state': 'CARRYING', 'event': 'confirmed',
                          'pad': round(pad, 2), 'opening': round(opening, 3)})

    idx, dist = carried_cookie(eef)
    allowed, why = release_ok(idx, eef) if st['state'] == 'CARRYING' else (True, {})
    n_blocked = 0
    if st['state'] == 'CARRYING' and st['cool'] == 0:
        for a in acts:
            if a[6] > 0:                       # a NEW open request
                if allowed:
                    st['releases'] += 1
                    st['state'] = 'NOT_CARRYING'
                    st['confirm'] = 0
                    st['log'].append({'frame': int(frame), 'event': 'release_allowed',
                                      'cookie': idx, 'why': why})
                else:
                    a[6] = -1.0
                    n_blocked += 1
        if n_blocked:
            st['blocked_rows'] += n_blocked
            st['blocked_events'] += 1
            st['consec_blocked'] += 1
            st['log'].append({'frame': int(frame), 'event': 'open_delayed', 'rows': n_blocked,
                              'cookie': idx, 'dist_m': round(dist, 3), 'why': why})
        if st['consec_blocked'] >= GATE_MAX_CONSEC:
            st['consec_blocked'] = 0
            st['cool'] = GATE_COOL
            st['log'].append({'frame': int(frame), 'event': 'cap_cool'})
    st['last_cmd_close'] = bool(acts[-1][6] < 0)
    meta['gate2'] = {'state': st['state'], 'blocked_rows': n_blocked, 'allowed': bool(allowed)}
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
finally:
    policy.chunk = real_chunk
    policy._infer_kwargs = real_infer_kwargs
    env.step = real_step
wall = round(time.time() - t0, 2)
rec, measured, chunks = D.serialize_result(result)
if hasattr(env, 'close'):
    env.close()

grip = np.zeros(len(store['opening']))
for k, row in enumerate(spy.rows):
    ad = row.get('adapter_out')
    if ad is None:
        continue
    ad = np.asarray(ad, np.float64).reshape(-1, 14)
    m = min(10, len(grip) - 10 * k)
    if m > 0:
        grip[10 * k:10 * k + m] = ad[:m, 6]
sg = np.where(grip >= 0, 1, -1)
m = measured.astype(np.float64)
summary = {**D.MARKERS, 'arm': 'G2', 'seed': SEED, 'wall_s': wall, 'record': rec,
           'state_machine': {'blocked_rows': st['blocked_rows'],
                             'open_requests_delayed': st['blocked_events'],
                             'releases_allowed': st['releases'],
                             'lost': st['lost'], 'force_aborts': st['aborts']},
           'executed_grip_flips': int((np.diff(sg) != 0).sum()),
           'force_absmax': float(np.abs(m).max()),
           'cookies_in_target': rec['cookies_in_target'],
           'cookies_in_source_final': next((c for c in reversed(store['src']) if c >= 0), -1),
           'pad_max': float(np.max(store['pad'])),
           'frames_pad_loaded': int(sum(1 for p in store['pad'] if p > PAD_LOAD))}
np.savez_compressed(OUT / ('G2_s%d.npz' % SEED), measured=measured,
                    step_eef_pose=np.asarray(store['eef'], np.float64),
                    step_opening=np.asarray(store['opening'], np.float64),
                    step_force18=np.asarray(store['force'], np.float64),
                    step_cookies=np.asarray(store['cookies'], np.int64),
                    step_cookies_in_source=np.asarray(store['src'], np.int64),
                    step_grip_cmd=grip, **D.MARKERS)
(OUT / ('G2_s%d.json' % SEED)).write_text(json.dumps(summary, indent=2, default=str))
(OUT / ('state_log_s%d.json' % SEED)).write_text(json.dumps(st['log'], indent=2, default=str))
print(json.dumps({k: summary[k] for k in ('wall_s', 'state_machine', 'executed_grip_flips',
                                          'force_absmax', 'cookies_in_target',
                                          'cookies_in_source_final', 'pad_max',
                                          'frames_pad_loaded')}), flush=True)
print('ALL_DONE', flush=True)