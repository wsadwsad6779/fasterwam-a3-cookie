"""State-matched supervision: for each recorded model chunk, what do the training labels do
at the nearest demonstration state?

For each of the 60 seed17 chunk decision frames we take the recorded proprio (8 dims), find
the nearest corpus row in z-scored proprio space, and compare
  - the model's gripper flip count over its 32 predicted rows, with
  - the label flip count over the 32 rows that follow the matched corpus row.

This asks the question "at a comparable state, is the model more chattery than its own
supervision?" without needing the images.

CPU only.
"""
import json

import numpy as np

BASE = '/workspace/eval_restore_20261117'
CORPUS = BASE + '/archive_evidence/data/a3_fw/arrays.npz'
FORE = BASE + '/reconstructed_normalizer/forensics_full_wide_out'
OUT = BASE + '/reconstructed_normalizer/GRIPPER_STATE_MATCHED.json'
H = 32

cz = np.load(FORE + '/seed17_original_chunks.npz')
chunks = np.stack([np.asarray(cz['chunk_%03d_model_out' % k], np.float64) for k in range(60)])
S = np.where(chunks[:, :, 6] >= 0, 1, -1)

az = np.load(CORPUS, allow_pickle=True)
prop = np.asarray(az['proprio'], np.float64)
act = np.asarray(az['action'], np.float64)
opening = np.asarray(az['opening'], np.float64)
ep_start = np.asarray(az['ep_start'], np.int64)
ep_len = np.asarray(az['ep_len'], np.int64)

# z-score in corpus space
mu = prop.mean(0)
sd = prop.std(0)
P = (prop - mu) / np.maximum(sd, 1e-9)

qz = np.load(FORE + '/seed17_observations.npz')
state = np.asarray(qz['observation.state'], np.float64)   # 14 dims at deployment

# Comparable quantities only: recorded eef position (eef_pose[0:3]) and measured opening
# (observation.state[7]); the corpus stores the same two things in proprio[:,0:3] and
# arrays['opening']. The 14-dim deployment state layout is not the 8-dim training layout,
# so nothing else is used here.
eef = np.asarray(qz['observation.eef_pose'], np.float64)
C = np.stack([prop[:, 0], prop[:, 1], prop[:, 2], opening], axis=1)
cmu = C.mean(0)
csd = np.maximum(C.std(0), 1e-9)
Cz = (C - cmu) / csd
rows = []
for k in range(60):
    f = 10 * k
    q = np.array([eef[f, 0], eef[f, 1], eef[f, 2], state[f, 7]])
    qq = (q - cmu) / csd
    d = np.linalg.norm(Cz - qq[None], axis=1)
    j = int(np.argmin(d))
    ep = int(np.searchsorted(ep_start, j, side='right') - 1)
    room = int(ep_start[ep] + ep_len[ep] - (j + H))
    if room < 0:
        fl = None
    else:
        seg = act[j:j + H, 6]
        fl = int((np.diff(seg) != 0).sum())
    rows.append({'chunk': k, 'frame': f, 'nn_row': j, 'nn_dist': float(d[j]),
                 'nn_episode': ep, 'nn_opening': float(opening[j]),
                 'model_flips': int((np.diff(S[k]) != 0).sum()),
                 'matched_label_flips_32': fl,
                 'query_xyz': q[:3].round(4).tolist(),
                 'nn_xyz': prop[j, :3].round(4).tolist()})

mf = np.array([r['model_flips'] for r in rows], float)
lf = np.array([(r['matched_label_flips_32'] if r['matched_label_flips_32'] is not None else np.nan)
               for r in rows], float)
dist = np.array([r['nn_dist'] for r in rows], float)
ok = ~np.isnan(lf)
res = {'NORMALIZER_IDENTITY': 'RECONSTRUCTED_DETERMINISTIC_NOT_ORIGINAL',
       'original_stats_sha256': 'UNKNOWN',
       'TRAINING_DATA_IDENTITY': 'CANDIDATE_RECONSTRUCTED_TRAINING_DATA',
       'match_space': '[eef x,y,z, measured opening] z-scored; nearest corpus row',
       'n_chunks': 60,
       'nn_distance': {'p50': float(np.percentile(dist, 50)), 'p90': float(np.percentile(dist, 90)),
                       'max': float(dist.max())},
       'model_flips_mean': float(mf.mean()),
       'matched_label_flips_mean': float(np.nanmean(lf)),
       'ratio': float(mf.mean() / max(np.nanmean(lf), 1e-9)),
       'model_flips_median': float(np.median(mf)),
       'matched_label_flips_median': float(np.nanmedian(lf)),
       'exact_state_neighbourhood': {
           'n_with_dist_lt_1': int((dist < 1.0).sum()),
           'model_flips_mean_dlt1': float(mf[dist < 1.0].mean()) if (dist < 1.0).any() else None,
           'label_flips_mean_dlt1': float(np.nanmean(lf[dist < 1.0])) if (dist < 1.0).any() else None},
       'per_chunk': rows}

with open(OUT, 'w') as fh:
    json.dump(res, fh, indent=2, default=str)
print(json.dumps({k: v for k, v in res.items() if k != 'per_chunk'}, indent=1, default=str))

# ---- kNN label distribution at the 9 sweep states ----
import os
SWEEP = BASE + '/reconstructed_normalizer/gpu_gripper_state_sweep_20261118/summary.json'
sweep_rows = []
if os.path.exists(SWEEP):
    sw = json.load(open(SWEEP))
    for rec in sw['results']:
        f = rec['frame']
        q = np.array([eef[f, 0], eef[f, 1], eef[f, 2], state[f, 7]])
        qq = (q - cmu) / csd
        d = np.linalg.norm(Cz - qq[None], axis=1)
        idx = np.argsort(d)[:20]
        vals, consts = [], []
        for j in idx:
            ep = int(np.searchsorted(ep_start, j, side='right') - 1)
            if ep_start[ep] + ep_len[ep] - (j + H) >= 0:
                seg = act[j:j + H, 6]
                vals.append(int((np.diff(seg) != 0).sum()))
                consts.append(int(vals[-1] == 0))
        sweep_rows.append({'frame': f, 'model_flips_mean': rec['flips_mean'],
                           'model_flips_min': rec['flips_min'], 'model_flips_max': rec['flips_max'],
                           'knn_label_flips_mean': float(np.mean(vals)) if vals else None,
                           'knn_label_const_fraction': float(np.mean(consts)) if consts else None,
                           'knn_dist_median': float(np.median(d[idx])),
                           'eef_z': rec['eef_z'], 'force_sigma': rec['force_max_abs_sigma'],
                           'opening': rec['opening']})
res['sweep_states'] = sweep_rows
with open(OUT, 'w') as fh:
    json.dump(res, fh, indent=2, default=str)
print('SWEEP_STATES')
for r in sweep_rows:
    print(r)

print('WROTE', OUT)