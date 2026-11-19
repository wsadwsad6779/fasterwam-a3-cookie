"""Prove or refute that the teacher corpus is one trajectory copied N times.

Checks, per archived seed:
  * poses / opening / 7-D action / force: bit equality against episode 0
  * first-window images: pixel equality at frames 0, 300, last, plus max abs diff
Output is small JSON (written next to the corpus).
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
P = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'teacher_replay_20261119'
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else P / 'diversity_audit.json'

arr = np.load(P / 'arrays.npz', allow_pickle=True)
pos = np.load(P / 'poses.npz')
forc = np.load(P / 'force_arrays.npz')
act = np.asarray(arr['action'])
pro = np.asarray(arr['proprio'])
f18 = np.asarray(forc['force18'])
st = np.asarray(arr['ep_start'])
ln = np.asarray(arr['ep_len'])
seeds = [int(k.split('_')[-1]) for k in pos.files if k.startswith('poses_')]

frames_path = P / 'frames.dat'
IMG = 224 * 448 * 3
mm = np.memmap(frames_path, dtype=np.uint8, mode='r').reshape(-1, 224, 448, 3)

ep = []
for k, seed in enumerate(seeds):
    s, n = int(st[k]), int(ln[k])
    poses = np.asarray(pos['poses_%d' % seed])
    opening = np.asarray(pos['opening_%d' % seed])
    f0 = int(s)
    idx = [0, min(300, n - 1), n - 1]
    imgs = {str(j): mm[f0 + j] for j in idx}
    ep.append({
        'k': k, 'seed': seed, 'start': s, 'len': n,
        'pose_shape': list(poses.shape), 'opening_shape': list(opening.shape),
        'action_sha': hashlib.sha256(act[s:s + n].tobytes()).hexdigest()[:16],
        'proprio_sha': hashlib.sha256(pro[s:s + n].tobytes()).hexdigest()[:16],
        'force_sha': hashlib.sha256(f18[s:s + n].tobytes()).hexdigest()[:16],
        'poses_sha': hashlib.sha256(poses.tobytes()).hexdigest()[:16],
        'opening_sha': hashlib.sha256(opening.tobytes()).hexdigest()[:16],
        'img_sha': {str(i): hashlib.sha256(imgs[str(i)].tobytes()).hexdigest()[:16]
                    for i in idx},
        '_poses': poses, '_opening': opening, '_action': act[s:s + n],
        '_imgs': imgs,
    })

ref = ep[0]
diffs = []
for e in ep[1:]:
    row = {'seed': e['seed'], 'vs_seed': ref['seed'], 'same_len': e['len'] == ref['len']}
    n = min(e['len'], ref['len'])
    row['pose_max_abs_diff'] = float(np.abs(e['_poses'][:n] - ref['_poses'][:n]).max())
    row['opening_max_abs_diff'] = float(np.abs(e['_opening'][:n] - ref['_opening'][:n]).max())
    row['action_max_abs_diff'] = float(np.abs(e['_action'][:n] - ref['_action'][:n]).max())
    s_e, s_r = int(st[e['k']]), int(st[ref['k']])
    row['force_max_abs_diff'] = float(np.abs(f18[s_e:s_e + n] - f18[s_r:s_r + n]).max())
    im = {}
    for key in ref['_imgs']:
        a, b = e['_imgs'][key], ref['_imgs'][key]
        im[key] = {'max': int(np.abs(a.astype(np.int16) - b.astype(np.int16)).max()),
                   'mean': float(np.abs(a.astype(np.float32) - b.astype(np.float32)).mean()),
                   'bit_equal': bool(np.array_equal(a, b))}
    row['images'] = im
    diffs.append(row)

out = {'dir': str(P), 'frames_on_disk': int(mm.shape[0]),
       'episodes': [{k: v for k, v in e.items() if not k.startswith('_')} for e in ep],
       'vs_first': diffs,
       'verdict_one_trajectory_copied': bool(
           all(d['pose_max_abs_diff'] == 0 and d['action_max_abs_diff'] == 0
               and all(v['bit_equal'] for v in d['images'].values()) for d in diffs))}
OUT.write_text(json.dumps(out, indent=1))
print('WROTE', OUT)
print(json.dumps({k: out[k] for k in ('frames_on_disk', 'verdict_one_trajectory_copied')}))