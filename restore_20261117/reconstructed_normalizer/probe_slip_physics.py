"""Physical verification of slip vs release vs never-grasped.

For every cookie the carry probe classified, this reads the six calibrated force channels
(wrist fx/fy/fz, pad_inner, pad_outer, gripper actuator) and classifies the let-go event:

  released_open         gripper command was open at let-go                 (intentional)
  confirmed_slip        pads loaded (>1.0 N) before let-go and both pads collapse (<0.3 N) at
                        let-go while the command stayed closed              (physical slip)
  collision_detach      wrist force magnitude at let-go exceeds 50 N or 5x its own pre-window  (impact)
  never_grasped         pads never exceeded 1.0 N in the pre-window          (no real grip)
  slip_unconfirmed      command closed, pads present but the collapse signature is incomplete

CPU only, reads the closed-loop npz.
"""
import glob
import json
import os

import numpy as np

BASE = '/workspace/eval_restore_20261117/reconstructed_normalizer'
CAL = json.load(open('/workspace/eval_restore_20261117/step_040000/force_calib_v1.json'))
CH = [int(c) for c in CAL['channels']]      # 0,1,2,12,13,16
DIRS = [BASE + '/closed_loop_scene_20261118_v2',
        BASE + '/closed_loop_gripper_fix_20261118',
        BASE + '/closed_loop_gripper_fix_20261118_v2',
        BASE + '/closed_loop_fix_s11', BASE + '/closed_loop_fix_s21']
OUT = BASE + '/SLIP_PHYSICS.json'


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


carry = json.load(open(BASE + '/CARRY_MODE.json'))
res = {'NORMALIZER_IDENTITY': 'RECONSTRUCTED_DETERMINISTIC_NOT_ORIGINAL',
       'original_stats_sha256': 'UNKNOWN',
       'thresholds': {'pad_loaded_N': 1.0, 'pad_collapse_N': 0.3,
                      'collision_force_N': 50.0, 'collision_ratio': 5.0},
       'arms': {}}

for arm, v in carry.items():
    d, f = arm.split('/')
    z = np.load(os.path.join(BASE, d, f + '.npz'))
    F = np.asarray(z['step_force18'], np.float64)[:, CH]
    pad = np.abs(F[:, [3, 4]]).max(1)      # pad_inner, pad_outer
    wrist = np.linalg.norm(F[:, 0:3], axis=1)
    g = grip_of(z)
    sign = np.where(g >= 0, 1, -1)
    rows = []
    for r in v['rows']:
        if not r.get('carried'):
            continue
        t = int(r['let_go_frame'])
        lo, hi = max(0, t - 10), t
        pre_pad = float(pad[lo:hi].max()) if hi > lo else float(pad[t])
        post_pad = float(pad[t:min(len(pad), t + 5)].min())
        pre_wrist = float(np.median(wrist[max(0, t - 10):hi])) if hi > lo else float(wrist[t])
        at_wrist = float(wrist[min(t, len(wrist) - 1)])
        mode = r['mode']
        if mode == 'released_open':
            cls = 'released_open'
        elif pre_wrist > 50.0 or (pre_wrist > 1e-6 and at_wrist > 5.0 * pre_wrist):
            cls = 'collision_detach'
        elif pre_pad <= 1.0:
            cls = 'never_grasped'
        elif post_pad < 0.3:
            cls = 'confirmed_slip'
        else:
            cls = 'slip_unconfirmed'
        rows.append({'cookie': r['cookie'], 'let_go': t, 'mode': mode, 'class': cls,
                     'pad_pre_max_N': round(pre_pad, 2), 'pad_post_min_N': round(post_pad, 2),
                     'wrist_pre_med_N': round(pre_wrist, 2), 'wrist_at_N': round(at_wrist, 2),
                     'xy_offset_mm': r.get('xy_offset_from_target_mm')})
    counts = {}
    for x in rows:
        counts[x['class']] = counts.get(x['class'], 0) + 1
    res['arms'][arm] = {'counts': counts, 'rows': rows}

with open(OUT, 'w') as fh:
    json.dump(res, fh, indent=2, default=str)
for k, v in res['arms'].items():
    print(k, json.dumps(v['counts']))
print('WROTE', OUT)