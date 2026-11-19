"""Release attitude: model vs expert.

Expert release attitude comes from proprio[:,3:6] (axis-angle of the EEF) at the label release
events. Model release attitude is the eef_pose quaternion (wxyz) at each 'released_open' let-go
frame found by probe_carry_mode.py. Reported: geodesic angle between each model release
rotation and the closest expert release rotation, and against the expert mean rotation.

CPU only.
"""
import json
import os

import numpy as np

BASE = '/workspace/eval_restore_20261117'
OUT = BASE + '/reconstructed_normalizer/RELEASE_ATTITUDE.json'


def aa_to_mat(v):
    th = np.linalg.norm(v)
    if th < 1e-12:
        return np.eye(3)
    k = v / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def quat_wxyz_to_mat(q):
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def geo(R1, R2):
    c = (np.trace(R1.T @ R2) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


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


# ---- expert ----
z = np.load(BASE + '/archive_evidence/data/a3_fw/arrays.npz', allow_pickle=True)
act = np.asarray(z['action'], np.float64)
prop = np.asarray(z['proprio'], np.float64)
es = np.asarray(z['ep_start'], np.int64)
el = np.asarray(z['ep_len'], np.int64)
expert_R = []
for s, L in zip(es, el):
    s, L = int(s), int(L)
    sign = act[s:s + L, 6]
    for (a, b, l) in runs(sign < 0):
        if l >= 10 and b + 1 < L and sign[b + 1] > 0:
            expert_R.append(aa_to_mat(prop[s + b + 1, 3:6]))
expert_R = np.stack(expert_R)
# expert mean rotation (project to SO(3))
M = expert_R.mean(0)
U, _, Vt = np.linalg.svd(M)
R_mean = U @ Vt
if np.linalg.det(R_mean) < 0:
    U[:, -1] *= -1
    R_mean = U @ Vt
res = {'NORMALIZER_IDENTITY': 'RECONSTRUCTED_DETERMINISTIC_NOT_ORIGINAL',
       'original_stats_sha256': 'UNKNOWN',
       'TRAINING_DATA_IDENTITY': 'CANDIDATE_RECONSTRUCTED_TRAINING_DATA',
       'expert_release_events': int(len(expert_R)),
       'arms': {}}

cm = json.load(open(BASE + '/reconstructed_normalizer/CARRY_MODE.json'))
for arm, v in cm.items():
    d = BASE + '/reconstructed_normalizer/' + arm.split('/')[0]
    f = arm.split('/')[1]
    zz = np.load(os.path.join(d, f + '.npz'))
    eefq = np.asarray(zz['step_eef_pose'], np.float64)
    rows = []
    for r in v['rows']:
        if r.get('mode') != 'released_open':
            continue
        t = int(r['let_go_frame'])
        R = quat_wxyz_to_mat(eefq[t, 3:7])
        dmean = geo(R, R_mean)
        dmin = min(geo(R, Re) for Re in expert_R)
        rows.append({'cookie': r['cookie'], 'frame': t,
                     'angle_to_expert_min_deg': round(dmin, 1),
                     'angle_to_expert_mean_deg': round(dmean, 1),
                     'xy_offset_mm': r['xy_offset_from_target_mm']})
    if rows:
        res['arms'][arm] = {
            'n': len(rows),
            'min_deg_p50': float(np.percentile([x['angle_to_expert_min_deg'] for x in rows], 50)),
            'mean_deg_p50': float(np.percentile([x['angle_to_expert_mean_deg'] for x in rows], 50)),
            'rows': rows}

with open(OUT, 'w') as fh:
    json.dump(res, fh, indent=2, default=str)
print(json.dumps({k: (v if k != 'arms' else {a: {kk: vv for kk, vv in b.items() if kk != 'rows'}
                                               for a, b in v.items()})
                  for k, v in res.items()}, indent=1, default=str)[:2500])
print('WROTE', OUT)