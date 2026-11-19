"""Release-gate G: hold the gripper closed until the EEF is inside a geometric release window.

Single change per step: if the model commands "open" while the EEF is outside the release
window, the command is replaced by "close". Everything else (model, weights, noise seeds,
images, force inputs, pm1, Adapter) is untouched.

Release window (from the env geometry):
  |x - 0.095| <= 0.045, |y - 0.100| <= 0.045, z <= 0.80
Protection:
  * at most GATE_MAX_CONSEC steps of continuous gating; afterwards the model command passes
    through for GATE_COOL steps (prevents endless closed-loop pushing)
  * if the measured wrist force exceeds 150 N, gating is suspended for the next 10 steps
Run per seed: baseline CL (no gate) already exists; this script runs only G.
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
SEED = int(os.environ.get('CL_GATE_SEED', '17'))
MAX_STEPS = int(os.environ.get('CL_MAX_STEPS', '600'))
REPLAN = 10
GATE_MAX_CONSEC = int(os.environ.get('GATE_MAX_CONSEC', '40'))
GATE_COOL = int(os.environ.get('GATE_COOL', '10'))
FORCE_SUSPEND_N = float(os.environ.get('GATE_FORCE_N', '150'))
OUT = ROOT / os.environ.get('CL_GATE_OUT', 'release_gate_20261118')
OUT.mkdir(exist_ok=False)
WINDOW = {'x': 0.045, 'y': 0.045, 'z': 0.80, 'cx': 0.095, 'cy': 0.100}

E, factory, W, fw = D.setup_protocol()
import torch
from a3_dual_arm_sim.contracts import STATE
from force_stream import CHANNELS, CHANNEL_NAMES
import record40k as R

policy, doc, calib, facts = D.build_policy_reconstructed(
    E, W, SimpleNamespace(cpu=False, gripper_mode='pm1'))
assert facts['device'] == 'cuda' and not policy.model.training
CLIP_LO = np.asarray(calib['per_channel_min'], dtype=np.float64)
CLIP_HI = np.asarray(calib['per_channel_max'], dtype=np.float64)

env = factory.make_env(position_noise_m=0.0, yaw_noise_rad=0.0, render_cameras=True)
reset = D.make_reset(E, factory, env, SimpleNamespace(
    cpu=False, gripper_mode='pm1', max_steps=MAX_STEPS, replan=REPLAN, allow_invalid_scene=False))

store = {k: [] for k in ('eef', 'opening', 'cookies', 'src', 'force', 'grip', 'gate')}
gate_log = []
gate_state = {'consec': 0, 'cool': 0, 'total_gated': 0, 'suspended': 0}


def snap(obs, info):
    store['eef'].append(np.asarray(obs['observation.eef_pose'], np.float64))
    store['opening'].append(float(np.asarray(obs[STATE])[7]))
    store['cookies'].append(int(info.get('cookies_in_target', -1)))
    store['src'].append(int(info.get('cookies_in_source', -1)))
    store['force'].append(np.asarray(obs['observation.force'], np.float64))


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


def gated_chunk(obs, frame, seed):
    acts, pf, meta = real_chunk(obs, frame, seed)
    eef = np.asarray(obs['observation.eef_pose'], np.float64)[:3]
    inside = (abs(eef[0] - WINDOW['cx']) <= WINDOW['x']
              and abs(eef[1] - WINDOW['cy']) <= WINDOW['y']
              and eef[2] <= WINDOW['z'])
    f = np.asarray(obs['observation.force'], np.float64)
    wrist = float(np.linalg.norm(f[[int(c) for c in CHANNELS][:3]]))
    n_gated = 0
    if gate_state['cool'] > 0:
        gate_state['cool'] -= 1
    elif wrist > FORCE_SUSPEND_N:
        gate_state['suspended'] += 1
        gate_state['consec'] = 0
        gate_state['cool'] = 10
    elif not inside:
        for a in acts:
            if a[6] > 0:                      # model wants to open
                a[6] = -1.0                   # hold closed
                n_gated += 1
        if n_gated:
            gate_state['consec'] += 1
            gate_state['total_gated'] += n_gated
        else:
            gate_state['consec'] = 0
        if gate_state['consec'] >= GATE_MAX_CONSEC:
            gate_state['consec'] = 0
            gate_state['cool'] = GATE_COOL
    else:
        gate_state['consec'] = 0
    gate_log.append({'frame': int(frame), 'eef': eef.round(4).tolist(), 'inside': bool(inside),
                     'gated_rows': int(n_gated), 'wrist_N': round(wrist, 2)})
    meta['gate'] = {'inside': bool(inside), 'gated_rows': int(n_gated)}
    return acts, pf, meta


policy.chunk = gated_chunk
spy = R.ActionSpy().install(policy)
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
t0 = time.time()
try:
    result = E.run_episode(env, policy, seed=SEED, reset=reset_and_snapshot,
                           max_steps=MAX_STEPS, replan=REPLAN)
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
summary = {**D.MARKERS, 'arm': 'G', 'seed': SEED, 'wall_s': wall, 'record': rec,
           'window': WINDOW, 'protection': {'max_consec': GATE_MAX_CONSEC, 'cool': GATE_COOL,
                                            'force_suspend_N': FORCE_SUSPEND_N},
           'gated_rows_total': int(gate_state['total_gated']),
           'installments_gated': int(sum(1 for g in gate_log if g['gated_rows'] > 0)),
           'force_suspensions': int(gate_state['suspended']),
           'executed_grip_flips': int((np.diff(sg) != 0).sum()),
           'force_absmax': float(np.abs(m).max()),
           'cookies_in_target': rec['cookies_in_target'],
           'cookies_in_source_final': next((c for c in reversed(store['src']) if c >= 0), -1),
           'frames_inside_window': int(sum(1 for g in gate_log if g['inside']))}
np.savez_compressed(OUT / ('G_s%d.npz' % SEED),
                    measured=measured,
                    step_eef_pose=np.asarray(store['eef'], np.float64),
                    step_opening=np.asarray(store['opening'], np.float64),
                    step_force18=np.asarray(store['force'], np.float64),
                    step_cookies=np.asarray(store['cookies'], np.int64),
                    step_cookies_in_source=np.asarray(store['src'], np.int64),
                    step_grip_cmd=grip,
                    **{('chunk_%03d_%s' % (i, k)): np.asarray(v)
                       for i, row in enumerate(spy.rows) for k, v in row.items()},
                    **D.MARKERS)
(OUT / ('G_s%d.json' % SEED)).write_text(json.dumps(summary, indent=2, default=str))
(OUT / ('gate_log_s%d.json' % SEED)).write_text(json.dumps(gate_log, indent=2, default=str))
print(json.dumps({k: summary[k] for k in ('wall_s', 'gated_rows_total', 'installments_gated',
                                          'force_suspensions', 'executed_grip_flips',
                                          'force_absmax', 'cookies_in_target',
                                          'cookies_in_source_final',
                                          'frames_inside_window')}), flush=True)
print('ALL_DONE', flush=True)