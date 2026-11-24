"""Record dual-view MuJoCo videos of the V3 continuous opening command and save
open/partial/closed frame snapshots for visual verification."""
import hashlib
import json
import os
import sys
import time

os.environ.setdefault('MUJOCO_GL', 'egl')
sys.path.insert(0, '/workspace/fwv1/sim_repo/src')

import imageio
import numpy as np

P = '/workspace/fwv1/p_items'
NPZ = P + '/A3_845_7D_LABELS_V3.npz'
OUTDIR = P + '/videos_v3'
OUT = P + '/GRIPPER_V3_REPLAY_VIDEO_MANIFEST.json'
EPISODES = [100, 422]


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 22), b''):
            h.update(c)
    return h.hexdigest()


def main():
    from a3_dual_arm_sim import A3CookieTransferEnv
    z = np.load(NPZ)
    L = z['label'].astype(np.float64)
    ls = z['ep_start_label'].astype(np.int64)
    el = z['ep_len'].astype(np.int64)
    os.makedirs(OUTDIR, exist_ok=True)
    entries = []
    for ep in EPISODES:
        s = int(ls[ep]); n = min(int(el[ep]), 600)
        cmd = L[s:s + n, 6]
        env = A3CookieTransferEnv(action_mode='cartesian_delta', render_mode='rgb_array', render_cameras=True)
        u = getattr(env, 'unwrapped', env)
        obs, _ = env.reset(seed=10000 + ep)
        frames_f, frames_w, opens = [], [], []
        snaps = {}
        try:
            for i in range(n):
                gcart = float(np.clip(2.0 * cmd[i] - 1.0, -1.0, 1.0))
                a = np.r_[0, 0, 0, 0, 0, 0, gcart, 0, 0, 0, 0, 0, 0, 1.0]
                obs, r, term, trunc, info = env.step(a)
                frames_f.append(np.ascontiguousarray(obs['observation.images.front']))
                frames_w.append(np.ascontiguousarray(obs['observation.images.left_wrist']))
                opens.append(float(u._gripper_state('L')[0]))
                c = cmd[i]
                key = 'open' if c >= 0.8 else ('partial' if c > 0.35 else 'closed')
                if key not in snaps:
                    snaps[key] = (len(frames_f) - 1, c)
                if term or trunc:
                    break
        finally:
            try:
                env.close()
            except Exception:
                pass
        vf = os.path.join(OUTDIR, 'v3_ep%04d_front.mp4' % ep)
        vw = os.path.join(OUTDIR, 'v3_ep%04d_left_wrist.mp4' % ep)
        imageio.mimsave(vf, frames_f, fps=20, codec='libx264', quality=8)
        imageio.mimsave(vw, frames_w, fps=20, codec='libx264', quality=8)
        snap_entries = []
        for key, (idx, c) in snaps.items():
            sp = os.path.join(OUTDIR, 'v3_ep%04d_%s_frame%04d.png' % (ep, key, idx))
            imageio.imwrite(sp, frames_f[idx])
            snap_entries.append({'state': key, 'frame': idx, 'cmd_opening': round(float(c), 3),
                                 'actual_opening': round(float(opens[idx]), 3), 'png': sp,
                                 'sha256': sha(sp)})
        entries.append({'episode': ep, 'frames': len(frames_f),
                        'front_mp4': vf, 'left_wrist_mp4': vw,
                        'front_sha256': sha(vf), 'left_wrist_sha256': sha(vw),
                        'snapshots': snap_entries,
                        'open_frac_cmd': float((cmd >= 0.8).mean()),
                        'reopen_count': int((np.diff((cmd >= 0.8).astype(int)) == 1).sum())})
    man = {'worker': 'task7', 'ts': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
           'dir': OUTDIR, 'episodes': entries,
           'note': 'continuous V3 opening command drives the real gripper; videos and open/partial/closed snapshots are direct visual evidence'}
    json.dump(man, open(OUT, 'w'), indent=1)
    print(json.dumps({'episodes': entries})[:1200])


if __name__ == '__main__':
    main()