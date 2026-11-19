"""Structural checks on the predicted gripper sequence.

  1. temporal median filter as an alternative post-hoc fix (vs hysteresis),
  2. position-mod-4 / mod-8 structure (action tokens are grouped 4 per video frame),
  3. supervision tail: how extreme can a 32-row label window get,
  4. per-row flip rate of the model split by group position.

CPU only.
"""
import json

import numpy as np

BASE = '/workspace/eval_restore_20261117'
FORE = BASE + '/reconstructed_normalizer/forensics_full_wide_out'
OUT = BASE + '/reconstructed_normalizer/GRIPPER_STRUCTURE.json'

cz = np.load(FORE + '/seed17_original_chunks.npz')
chunks = np.stack([np.asarray(cz['chunk_%03d_model_out' % k], np.float64) for k in range(60)])
V = chunks[:, :, 6]                      # (60, 32)
S = np.where(V >= 0, 1, -1)

az = np.load(BASE + '/archive_evidence/data/a3_fw/arrays.npz', allow_pickle=True)
act = np.asarray(az['action'], np.float64)
op = np.asarray(az['opening'], np.float64)
eps = np.asarray(az['ep_start'], np.int64)
eln = np.asarray(az['ep_len'], np.int64)

# --- 1. median filter ---
def med(v, k):
    pad = k // 2
    x = np.pad(v, pad, mode='edge')
    return np.array([np.median(x[i:i + k]) for i in range(len(v))])

res = {'NORMALIZER_IDENTITY': 'RECONSTRUCTED_DETERMINISTIC_NOT_ORIGINAL',
       'original_stats_sha256': 'UNKNOWN'}
mf = {}
for k in (1, 3, 5, 7):
    flips, exec_flips, intent = 0, 0, 0
    for i in range(60):
        v = med(V[i], k)
        s = np.where(v >= 0, 1, -1)
        flips += int((np.diff(s) != 0).sum())
        exec_flips += int((np.diff(s[:10]) != 0).sum())
        if (np.sign(v[:10]).sum() >= 0) != (S[i][:10].sum() >= 0):
            intent += 1
    mf[str(k)] = {'mean_flips_32': flips / 60.0, 'mean_flips_exec_10': exec_flips / 60.0,
                  'executed_majority_intent_changes': intent}
res['median_filter'] = mf

# --- 2. mod-4 / mod-8 position structure ---
res['position_structure'] = {}
for m in (4, 8):
    means = np.array([V[:, t::m].mean() for t in range(m)])
    overall = V.mean()
    res['position_structure']['mod%d_mean' % m] = means.round(4).tolist()
    res['position_structure']['mod%d_std' % m] = float(means.std())
    res['position_structure']['global_mean'] = float(overall)
    res['position_structure']['global_std'] = float(V.std())
    # flip rate at rows that are multiples of m vs others
    d = np.diff(S, axis=1) != 0
    idx = np.arange(1, 32)
    res['position_structure']['mod%d_flip_rate_at_boundaries' % m] = float(d[:, (idx % m) == 0].mean())
    res['position_structure']['mod%d_flip_rate_elsewhere' % m] = float(d[:, (idx % m) != 0].mean())
    # within-group spread of the value (rows inside one group)
    groups = V.reshape(60, 32 // m, m)
    res['position_structure']['within_group_value_std_mod%d' % m] = float(groups.std(axis=2).mean())
    res['position_structure']['across_group_value_std_mod%d' % m] = float(groups.mean(axis=2).std())

# --- 3. supervision tail ---
fl = []
for s, L in zip(eps, eln):
    s, L = int(s), int(L)
    g = act[s:s + L, 6]
    for w in range(L - 32):
        seg = g[w:w + 32]
        fl.append(int((np.diff(seg) != 0).sum()))
fl = np.asarray(fl)
res['supervision_tail'] = {'mean': float(fl.mean()), 'p50': float(np.percentile(fl, 50)),
                           'p90': float(np.percentile(fl, 90)), 'p99': float(np.percentile(fl, 99)),
                           'max': int(fl.max()),
                           'frac_ge_6': float((fl >= 6).mean()),
                           'frac_ge_10': float((fl >= 10).mean()),
                           'frac_ge_12': float((fl >= 12).mean()),
                           'n_windows': int(fl.size)}

# --- 4. model per-position flip rate and relation to group index ---
d = np.diff(S, axis=1) != 0
res['model_position_flip_rate'] = d.mean(0).round(4).tolist()
res['model_position_mean_value'] = V.mean(0).round(4).tolist()

with open(OUT, 'w') as fh:
    json.dump(res, fh, indent=2, default=str)
print(json.dumps(res, indent=1, default=str)[:4000])
print('WROTE', OUT)