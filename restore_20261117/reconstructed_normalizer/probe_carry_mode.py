"""Carry / release mode per moved cookie.

A cookie counts as carried if it comes within 5 cm of the EEF at some point after first
motion. The moment it leaves that neighbourhood for good is the "let go" frame; the gripper
sign at that frame separates a deliberate release (open) from a slip (still closed).
Also reports the EEF position at let-go and the vertical drop afterwards.
"""
import glob
import json
import os

import numpy as np

BASE = '/workspace/eval_restore_20261117/reconstructed_normalizer'
DIRS = [BASE + '/closed_loop_scene_20261118_v2',
        BASE + '/closed_loop_gripper_fix_20261118',
        BASE + '/closed_loop_gripper_fix_20261118_v2',
        BASE + '/closed_loop_fix_s11', BASE + '/closed_loop_fix_s21']
NEAR = 0.05
OUT = BASE + '/CARRY_MODE.json'
TARGET = np.array([0.095, 0.100])


def grip_of(z):
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


res = {}
for d in DIRS:
    if not os.path.isdir(d):
        continue
    for f in sorted(glob.glob(d + '/*.npz')):
        z = np.load(f)
        if 'step_cookie_pos' not in z.files:
            continue
        cp = np.asarray(z['step_cookie_pos'], np.float32)
        if cp.ndim != 3 or cp.shape[1] == 0 or np.isnan(cp).all():
            continue
        eef = np.asarray(z['step_eef_pose'], np.float64)[:, :3]
        g = grip_of(z)
        sign = np.where(g >= 0, 1, -1)
        T, C = cp.shape[0], cp.shape[1]
        base = cp[0]
        disp = np.linalg.norm(cp - base[None], axis=2).max(0)
        rows = []
        for c in range(C):
            if disp[c] <= 0.020:
                continue
            p = cp[:, c].astype(np.float64)
            dist = np.linalg.norm(p - eef, axis=1)
            near = dist < NEAR
            if not near.any():
                rows.append({'cookie': c, 'carried': False,
                             'min_dist_mm': round(float(dist.min() * 1000), 1),
                             'reason': 'never within 5 cm of the EEF'})
                continue
            last_near = int(np.flatnonzero(near)[-1])
            t_go = min(last_near + 1, T - 1)
            rows.append({
                'cookie': c, 'carried': True,
                'first_near': int(np.flatnonzero(near)[0]),
                'last_near': last_near,
                'let_go_frame': t_go,
                'grip_sign_at_let_go': int(sign[t_go]),
                'mode': 'released_open' if sign[t_go] > 0 else 'slip_closed',
                'eef_at_let_go': np.round(eef[t_go], 4).tolist(),
                'xy_offset_from_target_mm': round(float(np.linalg.norm(eef[t_go, :2] - TARGET) * 1000), 1),
                'cookie_speed_at_let_go_mms': round(float(np.linalg.norm(p[t_go] - p[max(0, t_go - 1)]) * 20 * 1000), 1),
                'final': np.round(p[-1], 4).tolist(),
                'drop_after_letgo_mm': round(float((p[last_near, 2] - p[-1, 2]) * 1000), 1)})
        modes = {}
        for r in rows:
            if not r.get('carried'):
                key = 'never_near_eef'
            else:
                key = r['mode']
            modes[key] = modes.get(key, 0) + 1
        res[os.path.basename(d) + '/' + os.path.basename(f)[:-4]] = {'rows': rows, 'modes': modes}

with open(OUT, 'w') as fh:
    json.dump(res, fh, indent=2, default=str)
for k, v in res.items():
    print(k, json.dumps(v['modes']))
print('WROTE', OUT)