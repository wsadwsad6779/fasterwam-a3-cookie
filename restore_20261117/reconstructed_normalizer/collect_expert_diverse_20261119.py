"""Collect successful expert demonstrations across translated cookie-batch layouts.

Why this exists: the archived teacher corpus is one trajectory copied N times. The eval
reset uses position_noise_m=0 / yaw_noise_rad=0, so `randomize_cookies=True` draws zero
noise and every seed renders the identical scene; the scripted expert is deterministic.
Diversity here comes from `env_factory.apply_batch_offset` (shared xy translation of the
whole 80-cookie grid), which the expert follows because it computes its pick centre from
`env.cookie_positions`.

Gates:
  * every kept episode must finish with cookies_in_target == 10 and exact_2x5_fill true
  * the (0,0) anchor, seed 41, must reproduce the archived corpus bit-for-bit in force and
    to <1e-9 in the reconstructed 7-D action (same physics path, same labels)
Output: teacher_diverse_20261119/{frames.dat, arrays.npz, force_arrays.npz, poses.npz,
manifest.json}; frames.dat is written incrementally so RAM stays flat.
"""
import json
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np

SIM_SRC = '/workspace/eval_restore_20261117/deploy/a3_dual_arm_sim/src'
if SIM_SRC not in sys.path:
    sys.path.insert(0, SIM_SRC)

import diag_reconstructed_40k as D

ROOT = Path(__file__).resolve().parent
# Frames go to the overlay disk: /workspace (98 GB loop) has ~6.4 GB free because an
# earlier run left 49 GB of full-model snapshots there. The small metadata is copied back
# to /workspace so the corpus identity survives the container.
OUT = Path(os.environ.get('DIV_OUT', '/root/repair_runs/teacher_diverse_20261119'))
META = ROOT / 'teacher_diverse_20261119'
ANCHOR = ROOT / 'teacher_replay_20261119'
MAX_STEPS = int(os.environ.get('DIV_STEPS', '600'))
EP_TIMEOUT = int(os.environ.get('DIV_EP_TIMEOUT', '480'))
IMG = 224

# Seed 41 everywhere: reset noise is zero, so the scene is fully determined by (dx, dy)
# and the seed is only a reproducible label (it is also the archived anchor episode).
DEFAULT_SPEC = [
    # dx, dy (m), split, seed; the collector rejects any candidate that does not finish
    # with 10 cookies in the target and exact_2x5_fill, so a wide candidate list is safe.
    [0.0, 0.0, 'train', 41],
    [0.005, 0.0, 'train', 41], [-0.005, 0.0, 'train', 41],
    [0.010, 0.0, 'train', 41], [-0.010, 0.0, 'train', 41],
    [0.0, 0.004, 'train', 41], [0.0, -0.004, 'train', 41],
    [0.0, 0.008, 'train', 41], [0.0, -0.008, 'train', 41],
    [0.005, 0.008, 'train', 41], [-0.005, -0.008, 'train', 41],
    [0.010, 0.004, 'train', 41], [-0.010, -0.004, 'train', 41],
    [0.0075, 0.002, 'val', 41], [-0.0075, -0.002, 'val', 41],
    [0.0025, 0.006, 'val', 41], [-0.0025, -0.006, 'val', 41],
]
SPEC = json.loads(os.environ['DIV_SPEC']) if os.environ.get('DIV_SPEC') else DEFAULT_SPEC

E, factory, W, fw = D.setup_protocol()
from a3_dual_arm_sim.controllers.batch_expert import A3CookieBatchExpert

calib = E.read_json(str(ROOT.parent / 'step_040000' / 'force_calib_v1.json'))


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
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def grip_labels(opening, eps=0.02, min_hold=3):
    L = len(opening)
    d = np.diff(opening)
    lab = np.empty(L)
    prev = 1.0
    for t in range(L - 1):
        if d[t] < -eps:
            prev = 1.0
        elif d[t] > eps:
            prev = -1.0
        lab[t] = prev
    lab[L - 1] = lab[L - 2] if L > 1 else 1.0
    for _ in range(3):
        i = 0
        while i < L:
            j = i
            while j < L and lab[j] == lab[i]:
                j += 1
            if j - i < min_hold:
                lab[i:j] = lab[i - 1] if i > 0 else lab[i]
            i = j
    return lab


class EpisodeTimeout(Exception):
    pass


def _on_alarm(_sig, _frm):
    raise EpisodeTimeout()


def run(spec):
    dx, dy, split, seed = float(spec[0]), float(spec[1]), str(spec[2]), int(spec[3])
    env = factory.make_env(position_noise_m=0.0, yaw_noise_rad=0.0, render_cameras=True)
    try:
        u = getattr(env, 'unwrapped', env)
        obs, _ = factory.reset(env, seed, randomize_cookies=True, randomize_boxes=False,
                               batch_offset_m=0.0)
        if dx or dy:
            factory.apply_batch_offset(u, 0.0, offset=np.asarray([dx, dy], np.float64))
            obs = u._observation()
        cookies0 = np.asarray(u.cookie_positions, np.float64).copy()
        expert = A3CookieBatchExpert(env=u, action_mode='joint_position')
        expert.reset(context=None)
        poses = [np.asarray(obs['observation.eef_pose'], np.float64)]
        ops = [float(np.asarray(obs['observation.state'])[7])]
        props = [E.build_proprio14(obs, W, calib, where='diverse')]
        f_step = [np.asarray(obs['observation.force'], np.float64)]
        # Frame t is the observation the action at step t is conditioned on, so the reset
        # image is frame 0 and the post-step image is written one iteration later.
        frames = []
        img_cur = np.concatenate([resize(obs['observation.images.front']),
                                  resize(obs['observation.images.left_wrist'])], axis=1)
        for _t in range(MAX_STEPS):
            a = expert.act(obs, '')
            out = env.step(np.asarray(a, np.float64))
            obs = out[0]
            frames.append(img_cur)
            img_cur = np.concatenate([resize(obs['observation.images.front']),
                                      resize(obs['observation.images.left_wrist'])], axis=1)
            poses.append(np.asarray(obs['observation.eef_pose'], np.float64))
            ops.append(float(np.asarray(obs['observation.state'])[7]))
            props.append(E.build_proprio14(obs, W, calib, where='diverse'))
            f_step.append(np.asarray(obs['observation.force'], np.float64))
        info = out[4] if len(out) > 4 else {}
    finally:
        if hasattr(env, 'close'):
            env.close()
    poses = np.asarray(poses)
    ops = np.asarray(ops)
    act = np.zeros((len(poses) - 1, 7))
    for t in range(len(poses) - 1):
        act[t, :3] = (poses[t + 1][:3] - poses[t][:3]) / 0.05
        act[t, 3:6] = rotvec_between(quat_wxyz_to_mat(poses[t][3:7]),
                                     quat_wxyz_to_mat(poses[t + 1][3:7])) / 0.5
    act[:, :6] = np.clip(act[:, :6], -1, 1)
    act[:, 6] = grip_labels(ops)[:len(act)]
    rec = {'dx': dx, 'dy': dy, 'split': split, 'seed': seed,
           'frames': int(len(act)),
           'cookies_in_target': int(info.get('cookies_in_target', -1)),
           'cookies_in_source': info.get('cookies_in_source'),
           'exact_fill': bool(info.get('exact_2x5_fill', False)),
           'expert_finished': bool(getattr(expert, 'finished', False)),
           'expert_phase': str(getattr(expert, 'phase', '')),
           'path_mm': round(float(np.linalg.norm(np.diff(poses[:, :3], axis=0),
                                                 axis=1).sum() * 1000), 2),
           'batch_center': cookies0[:5].mean(0).round(6).tolist(),
           'cookie_positions_sha': __import__('hashlib').sha256(
               cookies0.tobytes()).hexdigest()[:16]}
    print(json.dumps(rec), flush=True)
    return rec, dict(poses=poses, opening=ops[:-1].astype(np.float32),
                     proprio=np.asarray(props[:-1], np.float32),
                     force=np.asarray(f_step[:-1], np.float32), action=act.astype(np.float32),
                     cookies=cookies0, frames=np.stack(frames).astype(np.uint8))


def anchor_gate(rec, ep):
    old = np.load(ANCHOR / 'arrays.npz', allow_pickle=True)
    of = np.load(ANCHOR / 'force_arrays.npz', allow_pickle=True)
    old_a = np.asarray(old['action'], np.float64)
    old_f = np.asarray(of['force18'], np.float64)
    n = min(len(ep['action']), len(old_a))
    return {'force_max_abs_diff': float(np.abs(ep['force'][:n] - old_f[:n]).max()),
            'force_bit_equal': bool(np.array_equal(ep['force'][:n], old_f[:n])),
            'action_max_abs_diff': float(np.abs(ep['action'][:n] - old_a[:n]).max()),
            'opening_max_abs_diff': float(np.abs(ep['opening'][:n]
                                                 - np.load(ANCHOR / 'poses.npz')['opening_41'][:n]).max())}


def main():
    if OUT.exists():
        raise SystemExit('%s exists; refusing to overwrite' % OUT)
    OUT.mkdir()
    starts, lens, actions, propios, forces, poses_all, opens_all, recs = \
        [], [], [], [], [], [], [], []
    rejected = []
    offset = 0
    t0 = time.time()
    signal.signal(signal.SIGALRM, _on_alarm)
    with open(OUT / 'frames.dat', 'wb') as fh:
        for spec in SPEC:
            signal.alarm(EP_TIMEOUT)
            try:
                rec, ep = run(spec)
            except EpisodeTimeout:
                rec = {'dx': float(spec[0]), 'dy': float(spec[1]),
                       'split': str(spec[2]), 'seed': int(spec[3]),
                       'reason': 'wall_clock_timeout_%ds' % EP_TIMEOUT}
                ep = None
                rec['rejected'] = True
                rejected.append(rec)
                print('[reject] %s' % json.dumps(rec), flush=True)
                continue
            finally:
                signal.alarm(0)
            rec['wall_s'] = round(time.time() - t0, 1)
            if rec['cookies_in_target'] != 10 or not rec['exact_fill']:
                rec['rejected'] = True
                rejected.append(rec)
                print('[reject] %s' % json.dumps(rec), flush=True)
                continue
            fh.write(ep['frames'].tobytes())
            rec['rejected'] = False
            recs.append(rec)
            starts.append(offset)
            lens.append(int(rec['frames']))
            offset += int(rec['frames'])
            actions.append(ep['action'])
            propios.append(ep['proprio'])
            forces.append(ep['force'])
            poses_all.append(ep['poses'])
            opens_all.append(ep['opening'])
            rec['cookies'] = ep['cookies'].round(5).tolist()
    if not recs or recs[0]['dx'] != 0.0 or recs[0]['dy'] != 0.0:
        raise SystemExit('ANCHOR_MISSING: first kept episode must be the (0,0) anchor')

    np.savez_compressed(OUT / 'arrays.npz', action=np.concatenate(actions),
                        proprio=np.concatenate(propios),
                        ep_start=np.asarray(starts, np.int64),
                        ep_len=np.asarray(lens, np.int64),
                        EXPERT_ACTIONS=np.asarray(True), **D.MARKERS)
    cal = json.load(open(ROOT.parent / 'step_040000' / 'force_calib_v1.json'))
    fc = [int(c) for c in cal['channels']]
    F18 = np.concatenate(forces)
    np.savez_compressed(OUT / 'force_arrays.npz', force18=F18, force6=F18[:, fc],
                        ep_start=np.asarray(starts, np.int64),
                        ep_len=np.asarray(lens, np.int64))
    np.savez_compressed(OUT / 'poses.npz',
                        **{'poses_ep%02d' % i: p for i, p in enumerate(poses_all)},
                        **{'opening_ep%02d' % i: o for i, o in enumerate(opens_all)})

    gate = anchor_gate(recs[0], dict(force=forces[0], action=actions[0],
                                     opening=opens_all[0]))
    man = {**D.MARKERS, 'TRAINING_DATA_IDENTITY': 'CANDIDATE_RECONSTRUCTED_TRAINING_DATA',
           'TEACHER_DATA_IDENTITY': 'CANDIDATE_TEACHER_DATA',
           'diversity_axis': 'shared xy translation of the 80-cookie grid (batch_offset_m)',
           'why': 'eval reset noise is zero, so seeds do not change the scene; batch '
                  'translation is the only layout axis that actually moves the cookies',
           'episodes': recs, 'ep_start': starts, 'ep_len': lens, 'frames': int(offset),
           'frame_shape': [IMG, 2 * IMG, 3], 'bytes_frames': int(offset * IMG * 2 * IMG * 3),
           'train_episodes': sum(1 for r in recs if r['split'] == 'train'),
           'val_episodes': sum(1 for r in recs if r['split'] == 'val'),
           'rejected_candidates': rejected,
           'frames_location': str(OUT / 'frames.dat'),
           'frames_regenerable': 'deterministic: this script + manifest episodes (dx,dy,seed)',
           'all_full_task': True, 'anchor_gate': gate,
           'windows_num_frames33': int(sum(L - 33 + 1 for L in lens)),
           'wall_s': round(time.time() - t0, 1)}
    if man['val_episodes'] < 2 or man['train_episodes'] < 6:
        raise SystemExit('TOO_FEW_LAYOUTS train=%d val=%d'
                         % (man['train_episodes'], man['val_episodes']))
    (OUT / 'manifest.json').write_text(json.dumps(man, indent=2))
    import shutil
    META.mkdir(exist_ok=True)
    for name in ('arrays.npz', 'force_arrays.npz', 'poses.npz', 'manifest.json'):
        shutil.copy2(OUT / name, META / name)
    link = META / 'frames.dat'
    if not link.exists():
        os.symlink(str(OUT / 'frames.dat'), link)
    print(json.dumps({k: man[k] for k in ('episodes', 'frames', 'anchor_gate',
                                          'windows_num_frames33')}, indent=1)[:2500])
    print('DIVERSE_COLLECT_DONE', flush=True)


if __name__ == '__main__':
    main()