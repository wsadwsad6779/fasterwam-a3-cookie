"""Where do the plates go? Cookie-level outcome for every closed-loop arm.

Reads the npz files written by closed_loop_scene_20261118.py (seed 17/11/21, P2 clamp) and
closed_loop_gripper_fix_20261118(.v2) (CL/M5/H8 on seed 17). CPU only.

Reported per arm:
  executed gripper sign flips, one-step close pulses, close-run>=10 followed by open
  cookies_in_source start/final, cookies_in_target
  how many cookies moved >5 mm, >20 mm, their max displacement, max z reached
  how many end inside the target-bin footprint (centre 0.095/0.100, half 0.063/0.035)
  the displacement of the largest-moving cookie and its final position
"""
import json
import os

import numpy as np

BASE = '/workspace/eval_restore_20261117/reconstructed_normalizer'
DIRS = [BASE + '/closed_loop_scene_20261118_v2',
        BASE + '/closed_loop_gripper_fix_20261118',
        BASE + '/closed_loop_gripper_fix_20261118_v2']
TARGET = np.array([0.095, 0.100])
TARGET_HALF = np.array([0.063, 0.035])
OUT = BASE + '/COOKIE_OUTCOME_ALL_ARMS.json'


def runs(mask):
    out, i = [], 0
    while i < len(mask):
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < len(mask) and mask[j]:
            j += 1
        out.append((i, j - 1, j - i))
        i = j
    return out


def grip_stream(z):
    n = len(z['step_opening'])
    if 'step_grip_cmd' in z.files:
        return np.asarray(z['step_grip_cmd'], np.float64)
    g = np.zeros(n)
    for k in range(n // 10):
        key = 'chunk_%03d_adapter_out' % k
        if key not in z.files:
            break
        ad = np.asarray(z[key], np.float64).reshape(-1, 14)
        g[10 * k:10 * k + 10] = ad[:10, 6]
    return g


res = {'NORMALIZER_IDENTITY': 'RECONSTRUCTED_DETERMINISTIC_NOT_ORIGINAL',
       'original_stats_sha256': 'UNKNOWN', 'arms': {}}

for d in DIRS:
    if not os.path.isdir(d):
        continue
    for f in sorted(os.listdir(d)):
        if not f.endswith('.npz'):
            continue
        z = np.load(os.path.join(d, f), allow_pickle=False)
        if 'step_cookie_pos' not in z.files:
            continue
        cp = np.asarray(z['step_cookie_pos'], np.float32)
        if cp.ndim != 3 or cp.shape[1] == 0:
            continue
        g = grip_stream(z)
        sg = np.where(g >= 0, 1, -1)
        closed = sg < 0
        rs = runs(closed)
        releases = sum(1 for (s, e, l) in rs if l >= 10 and e + 1 < len(sg) and sg[e + 1] > 0)
        base = cp[0]
        disp = np.linalg.norm(cp - base[None], axis=2)
        maxd = disp.max(0)
        zmax = cp[:, :, 2].max(0)
        fin = cp[-1]
        in_fp = (np.abs(fin[:, 0] - TARGET[0]) <= TARGET_HALF[0]) & \
                (np.abs(fin[:, 1] - TARGET[1]) <= TARGET_HALF[1])
        biggest = int(np.argmax(maxd))
        rec = {
            'file': os.path.join(os.path.basename(d), f),
            'frames': int(cp.shape[0]), 'cookies': int(cp.shape[1]),
            'executed_grip_flips': int((np.diff(sg) != 0).sum()),
            'one_step_close_pulses': sum(1 for (_, _, l) in rs if l == 1),
            'short_close_pulses_le2': sum(1 for (_, _, l) in rs if l <= 2),
            'close_runs_ge10_followed_by_open': int(releases),
            'cookies_in_source_first': int(z['step_cookies_in_source'][0]) if 'step_cookies_in_source' in z.files else None,
            'cookies_in_source_final': int([c for c in z['step_cookies_in_source'] if c >= 0][-1]) if 'step_cookies_in_source' in z.files and any(c >= 0 for c in z['step_cookies_in_source']) else None,
            'cookies_in_target_final': int(z['step_cookies'][-1]) if 'step_cookies' in z.files else None,
            'moved_gt_5mm': int((maxd > 0.005).sum()),
            'moved_gt_20mm': int((maxd > 0.020).sum()),
            'moved_gt_50mm': int((maxd > 0.050).sum()),
            'max_displacement_mm': float(maxd.max() * 1000),
            'cookies_lifted_gt_20mm_z': int((zmax - base[:, 2] > 0.020).sum()),
            'final_in_target_footprint': int(in_fp.sum()),
            'biggest_cookie': {'id': biggest, 'disp_mm': float(maxd[biggest] * 1000),
                               'start': base[biggest].round(4).tolist(),
                               'end': fin[biggest].round(4).tolist(),
                               'z_max': float(zmax[biggest])},
            'end_xy_spread': {'x': [float(fin[:, 0].min()), float(fin[:, 0].max())],
                              'y': [float(fin[:, 1].min()), float(fin[:, 1].max())]},
        }
        res['arms'][os.path.join(os.path.basename(d), f)[:-4]] = rec

with open(OUT, 'w') as fh:
    json.dump(res, fh, indent=2, default=str)
print(json.dumps(res, indent=1, default=str)[:5000])
print('WROTE', OUT)