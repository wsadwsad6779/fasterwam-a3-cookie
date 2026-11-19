"""Expert replay collector (probe + full).

Uses the simulator's own expert controller (a3_dual_arm_sim.controllers.batch_expert) and rebuilds
the 7-dim action with the same recipe as prepare_a3_arrays.py:
    act[:3]  = diff(eef_position) / 0.05
    act[3:6] = rotvec(R_{t+1} R_t^T) / 0.5
    act[6]   = hysteretic grip label from diff(measured opening) (eps 0.02, min hold 3)
Only episodes with cookies_in_target >= 1 are kept.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

SIM_SRC = '/workspace/eval_restore_20261117/deploy/a3_dual_arm_sim/src'
if SIM_SRC not in sys.path:
    sys.path.insert(0, SIM_SRC)

import diag_reconstructed_40k as D

ROOT = Path(__file__).resolve().parent
OUT = ROOT / os.environ.get('EXPERT_OUT', 'teacher_replay_20261118')
SEEDS = [int(s) for s in os.environ.get('EXPERT_SEEDS', '41').split(',') if s.strip()]
PROBE = os.environ.get('EXPERT_PROBE', '1') == '1'
MAX_STEPS = int(os.environ.get('CL_MAX_STEPS', '600'))
IMG = 224
EPS, MIN_HOLD = 0.02, 3

E, factory, W, fw = D.setup_protocol()
from a3_dual_arm_sim.controllers.batch_expert import A3CookieBatchExpert


def resize(img):
    from PIL import Image
    a = np.asarray(img)
    if a.shape[0] == IMG and a.shape[1] == IMG:
        return a.astype(np.uint8)
    return np.asarray(Image.fromarray(a).resize((IMG, IMG), Image.BILINEAR), np.uint8)


def rotvec_between(Ra, Rb):
    R = Rb @ Ra.T
    c = np.clip((np.trace(R) - 1) / 2, -1, 1)
    ang = np.arccos(c)
    if ang < 1e-8:
        return np.zeros(3)
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return axis / (2 * np.sin(ang)) * ang


def quat_wxyz_to_mat(q):
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array([[1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
                     [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
                     [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)]])


def grip_labels(opening):
    L = len(opening)
    d = np.diff(opening)
    lab = np.empty(L)
    prev = 1.0
    for t in range(L - 1):
        if d[t] < -EPS:
            prev = 1.0
        elif d[t] > EPS:
            prev = -1.0
        lab[t] = prev
    lab[L - 1] = lab[L - 2] if L > 1 else 1.0
    for _ in range(3):
        i = 0
        while i < L:
            j = i
            while j < L and lab[j] == lab[i]:
                j += 1
            if j - i < MIN_HOLD:
                fill = lab[i - 1] if i > 0 else lab[i]
                lab[i:j] = fill
            i = j
    return lab


def run(seed):
    env = factory.make_env(position_noise_m=0.0, yaw_noise_rad=0.0, render_cameras=True)
    u = getattr(env, 'unwrapped', env)
    reset = D.make_reset(E, factory, env, type('A', (), {
        'cpu': False, 'gripper_mode': 'pm1', 'max_steps': MAX_STEPS, 'replan': 10,
        'allow_invalid_scene': False})())
    obs = reset(env, seed)
    expert = A3CookieBatchExpert(env=u, action_mode='joint_position')
    expert.reset(context=None)
    poses, ops, forces, imgs = [np.asarray(obs['observation.eef_pose'], np.float64)], \
                                [float(np.asarray(obs['observation.state'])[7])], \
                                [np.asarray(obs['observation.force'], np.float64)], \
                                [np.concatenate([resize(obs['observation.images.front']),
                                                 resize(obs['observation.images.left_wrist'])], axis=1)]
    info = {}
    for t in range(MAX_STEPS):
        a = expert.act(obs, '')
        out = env.step(np.asarray(a, np.float64))
        obs = out[0]
        info = out[4] if len(out) > 4 else {}
        poses.append(np.asarray(obs['observation.eef_pose'], np.float64))
        ops.append(float(np.asarray(obs['observation.state'])[7]))
        forces.append(np.asarray(obs['observation.force'], np.float64))
        imgs.append(np.concatenate([resize(obs['observation.images.front']),
                                    resize(obs['observation.images.left_wrist'])], axis=1))
    tgt = int(info.get('cookies_in_target', 0))
    print('[seed %d] steps=%d cookies_in_target=%d source=%s'
          % (seed, len(poses) - 1, tgt, info.get('cookies_in_source')), flush=True)
    if PROBE:
        env.close() if hasattr(env, 'close') else None
        return None
    ops = np.asarray(ops)
    act = np.zeros((len(poses) - 1, 7))
    for t in range(len(poses) - 1):
        act[t, :3] = (poses[t + 1][:3] - poses[t][:3]) / 0.05
        act[t, 3:6] = rotvec_between(quat_wxyz_to_mat(poses[t][3:7]),
                                     quat_wxyz_to_mat(poses[t + 1][3:7])) / 0.5
    act[:, :6] = np.clip(act[:, :6], -1, 1)
    act[:, 6] = grip_labels(ops)[:len(act)]
    env.close() if hasattr(env, 'close') else None
    return {'target': tgt, 'frames': np.stack(imgs[:-1]), 'action': act, 'poses': np.stack(poses),
            'opening': ops[:-1].astype(np.float32),
            'force18': np.stack(forces[:-1]),
            'state': np.asarray([np.asarray(obs['observation.state'])])}


if PROBE:
    for s in SEEDS:
        run(s)
    sys.exit(0)

if OUT.exists():
    raise SystemExit('refusing to overwrite %s' % OUT)
OUT.mkdir()
frames, actions, opens, forces18, starts, lens, kept = [], [], [], [], [], [], []
RECS = []
for s in SEEDS:
    r = run(s)
    if r is None or r['target'] < 1:
        print('[skip] seed %d target=%s' % (s, None if r is None else r['target']), flush=True)
        continue
    kept.append(s)
    starts.append(len(frames))
    lens.append(len(r['frames']))
    frames.append(r['frames'])
    actions.append(r['action'])
    opens.append(r['opening'])
    forces18.append(r['force18'])
    RECS.append(r)
if not kept:
    raise SystemExit('NO_SUCCESSFUL_EPISODES')
F = np.concatenate(frames).astype(np.uint8)
F.tofile(OUT / 'frames.dat')
op_cat = np.concatenate(opens)
prop = np.zeros((len(F), 14), np.float32)
for k, r in enumerate(RECS):
    s0, ln = starts[k], lens[k]
    P = r['poses'][:-1]
    prop[s0:s0 + ln, 0:3] = P[:, :3]
    for t in range(ln):
        prop[s0 + t, 3:6] = rotvec_between(np.eye(3), quat_wxyz_to_mat(P[t, 3:7]))
    prop[s0:s0 + ln, 6] = 0.04 * op_cat[s0:s0 + ln]
    prop[s0:s0 + ln, 7] = -0.04 * op_cat[s0:s0 + ln]
np.savez_compressed(OUT / 'arrays.npz', action=np.concatenate(actions).astype(np.float32),
                    proprio=prop, ep_start=np.asarray(starts, np.int64),
                    ep_len=np.asarray(lens, np.int64), EXPERT_ACTIONS=np.asarray(True),
                    **D.MARKERS)
FO = np.concatenate(forces18)
cal = json.load(open(ROOT.parent / 'step_040000' / 'force_calib_v1.json'))
fc = [int(c) for c in cal['channels']]
np.savez_compressed(OUT / 'force_arrays.npz', force18=FO.astype(np.float32),
                    force6=FO[:, fc].astype(np.float32))
man = {**D.MARKERS, 'seeds_kept': kept, 'episodes': len(kept), 'frames': int(F.shape[0]),
       'frame_shape': list(F.shape[1:]), 'bytes_frames': int(F.nbytes),
       'EXPERT_ACTIONS': True, 'action_dim': 7,
       'TEACHER_DATA_IDENTITY': 'CANDIDATE_TEACHER_DATA',
       'note': 'proprio xyz/aa columns are zero-filled here; the full 8/14-dim layout must be '
               'restored from eef_pose before training'}
(OUT / 'manifest.json').write_text(json.dumps(man, indent=2))
print(json.dumps(man, indent=1)[:900], flush=True)
print('REPLAY_DONE', flush=True)