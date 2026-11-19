"""Seed-21 release-event audit: for every open command G2 allowed, was a plate actually held,
where was it, and where did it end?

Joins state_log_s21.json (release_allowed entries with cookie id) with the npz (cookie poses,
pad forces, opening, grip). For each allowed release it reports the carried cookie's pad load
before the release, its bin-local geometry at release, its speed, and its final resting place
relative to the inner box.

CPU only.
"""
import json
import os

import numpy as np

BASE = '/workspace/eval_restore_20261117/reconstructed_normalizer'
D = BASE + '/g2_s21'
z = np.load(D + '/G2_s21.npz')
log = json.load(open(D + '/state_log_s21.json'))
f = np.asarray(z['step_force18'], np.float64)
op = np.asarray(z['step_opening'], np.float64)
g = np.asarray(z['step_grip_cmd'], np.float64)
eef = np.asarray(z['step_eef_pose'], np.float64)[:, :3]
cp = z['step_cookie_pos'] if 'step_cookie_pos' in z.files else None

TARGET = np.array([0.095, 0.100])
HALF = np.array([0.025, 0.00317, 0.0125])
FLOOR_TOP = 0.006
INNER = np.array([0.057, 0.029])
TOL = 0.004

releases = [x for x in log if x.get('event') == 'release_allowed']
print('allowed_releases', len(releases))
rows = []
for x in releases:
    fr = int(x['frame'])
    c = x.get('cookie')
    pad_before = float(max(abs(f[max(0, fr - 3):fr + 1, 12]).max(),
                           abs(f[max(0, fr - 3):fr + 1, 13]).max()))
    row = {'frame': fr, 'cookie': c, 'pad_before_N': round(pad_before, 2),
           'opening': round(float(op[min(fr, len(op) - 1)]), 3),
           'grip': round(float(g[min(fr, len(g) - 1)]), 2),
           'why': x.get('why')}
    if cp is not None and c is not None and c < cp.shape[1]:
        p0 = np.asarray(cp[min(fr, cp.shape[0] - 1), c], np.float64)
        row['cookie_xyz_at_release'] = p0.round(4).tolist()
    rows.append(row)

print(json.dumps(rows[:12], indent=1, default=str)[:2500])
if cp is not None:
    fin = np.asarray(cp[-1], np.float64)
    moved = np.linalg.norm(np.asarray(cp) - np.asarray(cp)[0][None], axis=2).max(0)
    near = np.flatnonzero(np.linalg.norm(fin[:, :2] - TARGET, axis=1) < 0.08)
    print('cookies_within_80mm_of_target_at_end', near.tolist())
    for j in near[:8]:
        print(' cookie', int(j), 'final', fin[j].round(4).tolist(),
              'moved_mm', round(float(moved[j] * 1000), 1))
json.dump({'allowed': rows}, open(BASE + '/G2_S21_RELEASE_AUDIT.json', 'w'), indent=2, default=str)
print('WROTE', BASE + '/G2_S21_RELEASE_AUDIT.json')