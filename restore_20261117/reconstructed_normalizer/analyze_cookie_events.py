"""Per-cookie placement event decomposition for the existing closed-loop arms.

For every cookie that the policy moved (>20 mm) it records
  first motion, first frame inside the bin's outer footprint, the last gripper open command
  before that, the actual detachment (cookie starts to separate from the EEF), speed at release,
  first rest, final pose, signed margins against the sim's three vertical/horizontal terms,
  and a single failure category.

Categories (mutually exclusive, checked in this order):
  never_moved              displacement <= 20 mm
  released_far_from_bin    gripper opened before the cookie ever entered the outer footprint
  still_held_at_end        gripper still closed at the end and cookie near the EEF
  in_footprint_but_high    horizontal test passes at the final frame, bottom not on the floor
  landed_outside           final frame fails the horizontal test
  pass                     all three terms pass at the final frame

CPU only; reads the npz written by closed_loop_scene_20261118.py and closed_loop_gripper_fix_*.
"""
import glob
import json
import os

import numpy as np

BASE = '/workspace/eval_restore_20261117/reconstructed_normalizer'
DIRS = [BASE + '/closed_loop_scene_20261118_v2',
        BASE + '/closed_loop_gripper_fix_20261118',
        BASE + '/closed_loop_gripper_fix_20261118_v2',
        BASE + '/closed_loop_fix_s11', BASE + '/closed_loop_fix_s21',
        BASE + '/closed_loop_placement_probe_20261118']
OUT = BASE + '/COOKIE_EVENTS.json'
TARGET = np.array([0.095, 0.100])
OUTER = np.array([0.063, 0.035])
INNER = np.array([0.057, 0.029])
TOL = 0.004
FLOOR_TOP = 0.006
HALF = np.array([0.025, 0.0031667, 0.0125])


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


res = {'NORMALIZER_IDENTITY': 'RECONSTRUCTED_DETERMINISTIC_NOT_ORIGINAL',
       'original_stats_sha256': 'UNKNOWN',
       'categories': ['never_moved', 'released_far_from_bin', 'still_held_at_end',
                      'in_footprint_but_high', 'landed_outside', 'pass'],
       'arms': {}}

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
        op = np.asarray(z['step_opening'], np.float64)
        g = grip_of(z)
        sign = np.where(g >= 0, 1, -1)          # +1 open, -1 close (adapter convention)
        T, C = cp.shape[0], cp.shape[1]
        vel = np.linalg.norm(np.diff(cp, axis=0), axis=2) * 20.0     # m/s at 20 Hz
        vel = np.vstack([vel, vel[-1:]])
        base = cp[0]
        disp = np.linalg.norm(cp - base[None], axis=2).max(0)
        rows = []
        for c in range(C):
            if disp[c] <= 0.020:
                rows.append({'cookie': c, 'category': 'never_moved',
                             'max_disp_mm': round(float(disp[c] * 1000), 1)})
                continue
            p = cp[:, c]
            in_outer = (np.abs(p[:, 0] - TARGET[0]) <= OUTER[0]) & \
                       (np.abs(p[:, 1] - TARGET[1]) <= OUTER[1])
            t_outer = int(np.argmax(in_outer)) if in_outer.any() else None
            moved = np.flatnonzero(np.linalg.norm(p - base[c][None], axis=1) > 0.005)
            t_move = int(moved[0]) if moved.size else 0
            opens = np.flatnonzero((sign[:-1] < 0) & (sign[1:] >= 0)) + 1
            opens = opens[opens >= t_move]
            t_rel = int(opens[-1]) if opens.size else None
            # detachment: first frame after t_rel where the cookie-EEF distance grows > 15 mm
            t_det = None
            if t_rel is not None:
                d0 = np.linalg.norm(p[t_rel] - eef[t_rel])
                for t in range(t_rel + 1, T):
                    if np.linalg.norm(p[t] - eef[t]) > d0 + 0.015:
                        t_det = t
                        break

            slip = None
            if t_det is not None:
                slip = {'grip_at_detach': int(sign[t_det]),
                        'opened_before_detach': bool(t_rel is not None and t_rel < t_det),
                        'frames_between_cmd_and_detach': int(t_det - t_rel) if t_rel is not None else None,
                        'speed_at_detach_mms': round(float(vel[t_det, c] * 1000), 1)}
            fin = p[-1]
            mx = (INNER[0] + TOL) - (abs(fin[0] - TARGET[0]) + HALF[0])
            my = (INNER[1] + TOL) - (abs(fin[1] - TARGET[1]) + HALF[1])
            bottom = fin[2] - (0.753 + 0.006) - HALF[2]
            mz_lo = bottom - (-0.003)
            mz_hi = (FLOOR_TOP + 0.004) - bottom
            fp = bool(mx >= 0 and my >= 0)
            z_ok = bool(bottom >= -0.003 and bottom <= 0.008)
            if t_rel is not None and t_outer is not None and t_rel < t_outer:
                cat = 'released_far_from_bin'
            elif sign[-1] < 0 and np.linalg.norm(fin - eef[-1]) < 0.05:
                cat = 'still_held_at_end'
            elif fp and z_ok:
                cat = 'pass'
            elif fp:
                cat = 'in_footprint_but_high'
            else:
                cat = 'landed_outside'
            rows.append({'cookie': c, 'category': cat,
                         'max_disp_mm': round(float(disp[c] * 1000), 1),
                         't_move': t_move, 't_in_outer_footprint': t_outer,
                         't_release_cmd': t_rel, 't_detach': t_det,
                         'speed_at_release_mms': round(float(vel[t_rel, c] * 1000), 1) if t_rel is not None else None,
                         'final': np.round(fin, 4).tolist(),
                         'final_speed_mms': round(float(vel[-1, c] * 1000), 1),
                         'margin_x_mm': round(float(mx * 1000), 1),
                         'margin_y_mm': round(float(my * 1000), 1),
                         'margin_bottom_lo_mm': round(float(mz_lo * 1000), 1),
                         'margin_bottom_hi_mm': round(float(mz_hi * 1000), 1),
                         'fp_pass': fp, 'bottom_on_floor_pass': z_ok, 'slip': slip})
        cats = {}
        for r in rows:
            cats[r['category']] = cats.get(r['category'], 0) + 1
        res['arms'][os.path.basename(d) + '/' + os.path.basename(f)[:-4]] = {
            'rows': rows, 'category_counts': cats}

with open(OUT, 'w') as fh:
    json.dump(res, fh, indent=2, default=str)
for k, v in res['arms'].items():
    print(k, json.dumps(v['category_counts']))
print('WROTE', OUT)