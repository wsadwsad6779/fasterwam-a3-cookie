"""Release-position analysis: where does the expert let go, and where does the policy?

CPU only.

Expert (corpus): release = a transition into the opening label (-1) that follows a run of at
least 10 closing rows (+1). Position = proprio[:,0:3] at that row.
Policy (closed-loop npz): same rule on the executed gripper command stream (adapter row 6),
position = step_eef_pose[:,0:3].

Target bin centre is (0.095, 0.100); the ten slots span x +/-0.028 and y +/-0.01267.
"""
import glob
import json
import os

import numpy as np

BASE = '/workspace/eval_restore_20261117'
TARGET = np.array([0.095, 0.100])
OUT = BASE + '/reconstructed_normalizer/RELEASE_POSITIONS.json'


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


def release_events(sign, pos):
    """sign: +1 = closed; return list of release events (row, xyz)."""
    ev = []
    closed = sign < 0
    for (s, e, l) in runs(closed):
        if l >= 10 and e + 1 < len(sign) and sign[e + 1] > 0:
            ev.append((e + 1, pos[e + 1]))
    return ev


res = {'NORMALIZER_IDENTITY': 'RECONSTRUCTED_DETERMINISTIC_NOT_ORIGINAL',
       'original_stats_sha256': 'UNKNOWN',
       'TRAINING_DATA_IDENTITY': 'CANDIDATE_RECONSTRUCTED_TRAINING_DATA',
       'target_center': TARGET.tolist(), 'arms': {}}

# ---------- expert ----------
z = np.load(BASE + '/archive_evidence/data/a3_fw/arrays.npz', allow_pickle=True)
act = np.asarray(z['action'], np.float64)          # dim 6: +1 close, -1 open
prop = np.asarray(z['proprio'], np.float64)
es = np.asarray(z['ep_start'], np.int64)
el = np.asarray(z['ep_len'], np.int64)

offsets = []
per_ep = []
for s, L in zip(es, el):
    s, L = int(s), int(L)
    sign = act[s:s + L, 6]
    pos = prop[s:s + L, :3]
    ev = release_events(sign, pos)
    d = [float(np.linalg.norm(p[:2] - TARGET)) for _, p in ev]
    offsets += d
    per_ep.append({'ep': int(s), 'releases': len(ev),
                   'offsets_mm': np.round(np.array(d) * 1000, 1).tolist() if d else []})
offsets = np.asarray(offsets)
res['expert'] = {
    'release_events': int(offsets.size),
    'releases_per_episode_mean': float(np.mean([r['releases'] for r in per_ep])),
    'offset_mm': {'p50': float(np.percentile(offsets, 50)) if offsets.size else None,
                  'p90': float(np.percentile(offsets, 90)) if offsets.size else None,
                  'max': float(offsets.max()) if offsets.size else None,
                  'frac_le_15mm': float((offsets <= 0.015).mean()) if offsets.size else None,
                  'frac_le_30mm': float((offsets <= 0.030).mean()) if offsets.size else None},
    'per_episode_head': per_ep[:10],
}

# ---------- policy arms ----------
for d in [BASE + '/reconstructed_normalizer/closed_loop_scene_20261118_v2',
          BASE + '/reconstructed_normalizer/closed_loop_gripper_fix_20261118',
          BASE + '/reconstructed_normalizer/closed_loop_gripper_fix_20261118_v2']:
    if not os.path.isdir(d):
        continue
    for f in sorted(glob.glob(d + '/*.npz')):
        zz = np.load(f)
        if 'step_eef_pose' not in zz.files:
            continue
        n = len(zz['step_opening'])
        if 'step_grip_cmd' in zz.files:
            g = np.asarray(zz['step_grip_cmd'], np.float64)
        else:
            g = np.zeros(n)
            for k in range(n // 10):
                key = 'chunk_%03d_adapter_out' % k
                if key not in zz.files:
                    break
                ad = np.asarray(zz[key], np.float64).reshape(-1, 14)
                g[10 * k:10 * k + 10] = ad[:10, 6]
        sign = np.where(g >= 0, 1, -1)
        pos = np.asarray(zz['step_eef_pose'], np.float64)[:, :3]
        ev = release_events(sign, pos)
        doff = [float(np.linalg.norm(p[:2] - TARGET) * 1000) for _, p in ev]
        res['arms'][os.path.basename(d) + '/' + os.path.basename(f)[:-4]] = {
            'releases': len(ev),
            'release_frames': [int(r) for r, _ in ev],
            'offsets_mm': np.round(doff, 1).tolist() if doff else [],
            'offset_median_mm': float(np.median(doff)) if doff else None,
            'release_z': [round(float(p[2]), 4) for _, p in ev],
        }

with open(OUT, 'w') as fh:
    json.dump(res, fh, indent=2, default=str)
print(json.dumps({k: res[k] for k in ('expert', 'arms')}, indent=1, default=str)[:4000])
print('WROTE', OUT)