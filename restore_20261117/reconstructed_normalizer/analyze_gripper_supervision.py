"""Supervision vs generation: gripper (action dim 7) sign-flip statistics.

CPU only. Runs on the pod where the corpus and the recorded model outputs live.

  A. training supervision   arrays.npz action[:,6] rebuilt from measured opening per
                            prepare_a3_arrays.py (GRIP_EPS=1e-4, hold-previous, last row copy)
  B. model generation       seed17 recorded chunk model_out (60 chunks) and the v6 GPU jobs
  C. comparison             first-10 / full-32 flip rates, per-position flip probability,
                            replan-boundary rewrites, value magnitude near zero

Nothing here executes control or modifies any asset.
"""
import glob
import json
import os

import numpy as np

BASE = '/workspace/eval_restore_20261117'
CORPUS = BASE + '/archive_evidence/data/a3_fw/arrays.npz'
FORENSICS = BASE + '/reconstructed_normalizer/forensics_full_wide_out/seed17_original_chunks.npz'
V6 = BASE + '/reconstructed_normalizer/gpu_mechanism_v6_20261118'
OUT = BASE + '/reconstructed_normalizer/GRIPPER_SUPERVISION_VS_GENERATION.json'
EPS = 1e-4
NUM_FRAMES = 33
H = NUM_FRAMES - 1


def sign_pm1(x):
    return np.where(np.asarray(x, np.float64) >= 0.0, 1, -1)


def rebuild_labels(op):
    L = len(op)
    d = np.diff(op)
    g = np.full(L, -1.0, np.float32)
    for t in range(L - 1):
        if d[t] < -EPS:
            g[t] = 1.0
        elif d[t] > EPS:
            g[t] = -1.0
        else:
            g[t] = g[t - 1] if t > 0 else -1.0
    g[L - 1] = g[L - 2] if L > 1 else -1.0
    return g, d


def flip_stats_rows(rows, n_first=10):
    """rows: (..., H) already +/-1. Returns flips in first n_first and in all H."""
    seg = np.asarray(rows)
    d = np.diff(seg, axis=-1)
    return (d[..., :n_first - 1] != 0).sum(-1), (d != 0).sum(-1)


res = {'NORMALIZER_IDENTITY': 'RECONSTRUCTED_DETERMINISTIC_NOT_ORIGINAL',
       'original_stats_sha256': 'UNKNOWN',
       'TRAINING_DATA_IDENTITY': 'CANDIDATE_RECONSTRUCTED_TRAINING_DATA',
       'supervision_source': 'finetune/prepare_a3_arrays.py: action[:,6] = sign of measured '
                             'opening diff with GRIP_EPS=1e-4 deadband, hold-previous, '
                             'closing->+1, opening->-1, last row copied',
       'corpus_sha256': None}

z = np.load(CORPUS, allow_pickle=True)
act = np.asarray(z['action'], np.float64)
op = np.asarray(z['opening'], np.float64)
jc = np.asarray(z['joint_cmd'], np.float64)
ep_start = np.asarray(z['ep_start'], np.int64)
ep_len = np.asarray(z['ep_len'], np.int64)
res['corpus'] = {'rows': int(len(act)), 'episodes': int(len(ep_start)),
                 'ep_len_min': int(ep_len.min()), 'ep_len_mean': float(ep_len.mean()),
                 'ep_len_max': int(ep_len.max()),
                 'opening_min': float(op.min()), 'opening_max': float(op.max()),
                 'joint_cmd_min': float(jc.min()), 'joint_cmd_max': float(jc.max()),
                 'joint_cmd_unique_le8': np.unique(np.round(jc, 6))[:8].tolist()}

# A. rule reproduction + per-episode label structure
mism = 0
tot = 0
flips_per_ep = []
dmag = []
runlens = []
per_pos_hits = np.zeros(H)
per_pos_pairs = np.zeros(H)
f10_all = []
f32_all = []
nwin = 0
allrows = []
for s, L in zip(ep_start, ep_len):
    s, L = int(s), int(L)
    g, d = rebuild_labels(op[s:s + L])
    mism += int((g != act[s:s + L, 6]).sum())
    tot += L
    tr = np.flatnonzero(np.diff(g) != 0)
    flips_per_ep.append(int(len(tr)))
    if len(tr):
        dmag.append(np.abs(d[tr]))
    bounds = [-1] + tr.tolist() + [L - 1]
    runlens += np.diff(bounds).tolist()
    for w in range(L - H):
        seg = g[w:w + H]
        a10, a32 = flip_stats_rows(seg)
        f10_all.append(int(a10))
        f32_all.append(int(a32))
        for t in range(1, H):
            per_pos_pairs[t] += 1
            per_pos_hits[t] += int(seg[t] != seg[t - 1])
        nwin += 1
    allrows.append(g)

dmag = np.concatenate(dmag) if dmag else np.zeros(0)
f10_all = np.asarray(f10_all)
f32_all = np.asarray(f32_all)
runlens = np.asarray(runlens)
res['supervision'] = {
    'rule_reproduction_mismatch_rows': int(mism), 'rule_reproduction_rows': int(tot),
    'rule_reproduction_mismatch_fraction': float(mism / max(tot, 1)),
    'positive_fraction': float((act[:, 6] > 0).mean()),
    'flips_per_episode': {'mean': float(np.mean(flips_per_ep)), 'median': float(np.median(flips_per_ep)),
                          'max': int(np.max(flips_per_ep)), 'min': int(np.min(flips_per_ep)),
                          'per_100_frames_mean': float(np.mean(flips_per_ep) / np.mean(ep_len) * 100)},
    'run_length_frames': {'p50': float(np.percentile(runlens, 50)),
                          'p90': float(np.percentile(runlens, 90)),
                          'p99': float(np.percentile(runlens, 99)),
                          'max': int(runlens.max())},
    'flip_dmag': {'p50': float(np.percentile(dmag, 50)), 'p90': float(np.percentile(dmag, 90)),
                  'p99': float(np.percentile(dmag, 99)),
                  'frac_lt_1e_3': float((dmag < 1e-3).mean()),
                  'frac_lt_1e_2': float((dmag < 1e-2).mean()),
                  'frac_lt_5e_2': float((dmag < 5e-2).mean())},
    'windows': {'n': int(nwin),
                'mean_flips_first10_of_9': float(f10_all.mean()),
                'mean_flips_32_of_31': float(f32_all.mean()),
                'p_any_flip_first10': float((f10_all > 0).mean()),
                'p_any_flip_32': float((f32_all > 0).mean()),
                'p_constant_32': float((f32_all == 0).mean()),
                'f10_hist': {str(k): float((f10_all == k).mean()) for k in range(0, 6)},
                'f32_hist': {str(k): float((f32_all == k).mean()) for k in range(0, 9)}},
    'per_position_flip_rate': (per_pos_hits / np.maximum(per_pos_pairs, 1)).round(5).tolist(),
}

# B. model generation: seed17 recorded 60 chunks
fz = np.load(FORENSICS)
keys = ['chunk_%03d_model_out' % k for k in range(60)]
chunks = [np.asarray(fz[k], np.float64).reshape(H, 7) for k in keys]
sgn = np.stack([sign_pm1(c[:, 6]) for c in chunks])          # (60, 32)
raw = np.stack([c[:, 6] for c in chunks])                    # (60, 32) normalised values
f10, f32 = flip_stats_rows(sgn)
# replan boundary: executed row 9 (frame 10k+9) vs next chunk row 0 (frame 10k+10)
bnd = (sgn[1:, 0] != sgn[:-1, 9])
# within-chunk executed sequence flip rate (rows 0..9, 9 pairs)
exec_pairs = (np.diff(sgn[:, :10], axis=1) != 0)
res['generation_seed17_60chunks'] = {
    'mean_flips_first10_of_9': float(f10.mean()),
    'mean_flips_32_of_31': float(f32.mean()),
    'p_any_flip_first10': float((f10 > 0).mean()),
    'p_any_flip_32': float((f32 > 0).mean()),
    'p_constant_32': float((f32 == 0).mean()),
    'f10_hist': {str(k): float((f10 == k).mean()) for k in range(0, 6)},
    'f32_hist': {str(k): float((f32 == k).mean()) for k in range(0, 9)},
    'per_position_flip_rate': (np.diff(sgn, axis=1) != 0).mean(0).round(5).tolist(),
    'replan_boundary_flip_rate': float(bnd.mean()),
    'replan_boundary_pairs': int(bnd.size),
    'executed_row_flip_rate_rows0_9': (exec_pairs.mean(0)).round(5).tolist(),
    'raw_abs_lt_0_2_fraction': float((np.abs(raw) < 0.2).mean()),
    'raw_abs_lt_0_1_fraction': float((np.abs(raw) < 0.1).mean()),
    'raw_absmax': float(np.abs(raw).max()),
}

# C. v6 A0 jobs (7 real replan states, protocol noise)
v6 = {}
for f in sorted(glob.glob(V6 + '/*.npz')):
    name = os.path.basename(f)[:-4]
    if not name.endswith('A0') and 'repeat' not in name:
        continue
    d = np.load(f)
    if 'raw32' not in d.files:
        continue
    r = np.asarray(d['raw32'], np.float64)[:, 6]
    s = sign_pm1(r)
    a10, a32 = flip_stats_rows(s)
    v6[name] = {'flips_first10': int(a10), 'flips_32': int(a32),
                'sign_rows0_9': s[:10].tolist(),
                'absmax': float(np.abs(r).max()),
                'abs_lt_0_2_fraction': float((np.abs(r) < 0.2).mean())}
res['generation_v6_A0'] = v6

# D. side-by-side summary
res['comparison'] = {
    'per_pair_flip_rate': {
        'supervision_first10': res['supervision']['windows']['mean_flips_first10_of_9'] / 9.0,
        'model_seed17_first10': res['generation_seed17_60chunks']['mean_flips_first10_of_9'] / 9.0,
        'supervision_32': res['supervision']['windows']['mean_flips_32_of_31'] / 31.0,
        'model_seed17_32': res['generation_seed17_60chunks']['mean_flips_32_of_31'] / 31.0,
    },
    'caveat': 'supervision rows are exact +/-1 labels; model rows are thresholded at 0 on the '
              'normalised action. Identical statistics, different quantity types.',
}

with open(OUT, 'w') as fh:
    json.dump(res, fh, indent=2, default=str)

print(json.dumps({k: res[k] for k in ('corpus', 'supervision', 'generation_seed17_60chunks',
                                      'comparison', 'generation_v6_A0')}, indent=1, default=str)[:7000])
print('WROTE', OUT)