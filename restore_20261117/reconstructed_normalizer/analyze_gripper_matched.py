"""Matched-phase comparison of gripper sign patterns: supervision vs generation.

CPU only. Extends analyze_gripper_supervision.py with
  1. pattern classes (constant / single step / multi step / alternating) and run lengths,
  2. flip rate conditioned on the local opening phase (|d opening| at the decision frame),
  3. executed rows (0..9) vs planned-only rows (10..31) for the model,
  4. flip presence against measured force at the decision state,
  5. drift: flip rate by chunk index.

Outputs GRIPPER_MATCHED_PHASE.json next to the previous analysis.
"""
import json
import os

import numpy as np

BASE = '/workspace/eval_restore_20261117'
CORPUS = BASE + '/archive_evidence/data/a3_fw/arrays.npz'
FORENSICS = BASE + '/reconstructed_normalizer/forensics_full_wide_out'
OUT = BASE + '/reconstructed_normalizer/GRIPPER_MATCHED_PHASE.json'
EPS = 1e-4
H = 32


def sign_pm1(x):
    return np.where(np.asarray(x, np.float64) >= 0.0, 1, -1)


def classify(s):
    """s: +/-1 sequence. Returns (n_flips, class, min_run, n_single_runs)."""
    d = np.diff(s)
    n = int((d != 0).sum())
    runs = []
    cur = 1
    for t in range(1, len(s)):
        if s[t] == s[t - 1]:
            cur += 1
        else:
            runs.append(cur)
            cur = 1
    runs.append(cur)
    singles = int(sum(1 for r in runs if r == 1))
    if n == 0:
        cls = 'constant'
    elif n == 1:
        cls = 'single_step'
    elif singles >= max(1, len(runs) - 2):
        cls = 'alternating'
    else:
        cls = 'multi_step'
    return n, cls, min(runs), singles


def phase_bucket(dprev):
    a = abs(dprev)
    if a < 1e-3:
        return 'static'
    if a < 1e-2:
        return 'micro'
    return 'moving'


res = {'NORMALIZER_IDENTITY': 'RECONSTRUCTED_DETERMINISTIC_NOT_ORIGINAL',
       'original_stats_sha256': 'UNKNOWN',
       'TRAINING_DATA_IDENTITY': 'CANDIDATE_RECONSTRUCTED_TRAINING_DATA'}

# ---------- supervision ----------
z = np.load(CORPUS, allow_pickle=True)
act = np.asarray(z['action'], np.float64)
op = np.asarray(z['opening'], np.float64)
ep_start = np.asarray(z['ep_start'], np.int64)
ep_len = np.asarray(z['ep_len'], np.int64)

sup_cls = {}
sup_phase = {}
sup_flips = []
run_len_all = []
for s, L in zip(ep_start, ep_len):
    s, L = int(s), int(L)
    g = act[s:s + L, 6]
    bounds = [-1] + np.flatnonzero(np.diff(g) != 0).tolist() + [L - 1]
    run_len_all += np.diff(bounds).tolist()
    for w in range(L - H):
        seg = g[w:w + H]
        n, cls, minrun, singles = classify(seg)
        sup_cls[cls] = sup_cls.get(cls, 0) + 1
        sup_flips.append(n)
        b = phase_bucket(op[w] - op[w - 1] if w > 0 else 0.0)
        e = sup_phase.setdefault(b, {'n': 0, 'flips': 0, 'any': 0})
        e['n'] += 1
        e['flips'] += n
        e['any'] += int(n > 0)

sup_flips = np.asarray(sup_flips)
nwin = len(sup_flips)
res['supervision'] = {
    'windows': nwin,
    'pattern_class_fraction': {k: v / nwin for k, v in sorted(sup_cls.items())},
    'mean_flips_32': float(sup_flips.mean()),
    'run_length_frames_all': {'p50': float(np.percentile(run_len_all, 50)),
                              'p90': float(np.percentile(run_len_all, 90)),
                              'frac_run_eq_1': float((np.asarray(run_len_all) == 1).mean())},
    'by_phase': {k: {'n': v['n'], 'mean_flips': v['flips'] / v['n'],
                     'p_any_flip': v['any'] / v['n']} for k, v in sup_phase.items()},
}

# ---------- generation (seed17 recorded 60 chunks) ----------
cz = np.load(FORENSICS + '/seed17_original_chunks.npz')
chunks = np.stack([np.asarray(cz['chunk_%03d_model_out' % k], np.float64) for k in range(60)])
sg = sign_pm1(chunks[:, :, 6])                        # (60, 32)
vals = chunks[:, :, 6]
oz = np.load(FORENSICS + '/seed17_observations.npz')
state = np.asarray(oz['observation.state'], np.float64)
force18 = np.asarray(oz['observation.force'], np.float64)
calib = json.load(open(BASE + '/step_040000/force_calib_v1.json'))
ch = [int(c) for c in calib['channels']]
opening_obs = state[:, 7]
fmax_obs = np.abs(force18[:, ch]).max(1)

gen_cls = {}
gen_flips = []
gen_phase = {}
exec_flips = []
plan_flips = []
per_chunk = []
for k in range(60):
    s = sg[k]
    n, cls, minrun, singles = classify(s)
    gen_cls[cls] = gen_cls.get(cls, 0) + 1
    gen_flips.append(n)
    f = 10 * k
    dprev = float(opening_obs[f] - opening_obs[f - 1]) if f > 0 else 0.0
    b = phase_bucket(dprev)
    e = gen_phase.setdefault(b, {'n': 0, 'flips': 0, 'any': 0})
    e['n'] += 1
    e['flips'] += n
    e['any'] += int(n > 0)
    exec_flips.append(int((np.diff(s[:10]) != 0).sum()))
    plan_flips.append(int((np.diff(s[10:]) != 0).sum()))
    per_chunk.append({'chunk': k, 'frame': f, 'flips': n, 'class': cls,
                      'dprev': dprev, 'opening': float(opening_obs[f]),
                      'force_absmax': float(fmax_obs[f]),
                      'sign10': ''.join('+' if x > 0 else '-' for x in s[:10]),
                      'margin_absmin': float(np.abs(vals[k][:10]).min())})

gen_flips = np.asarray(gen_flips)
res['generation_seed17'] = {
    'chunks': 60,
    'pattern_class_fraction': {k: v / 60 for k, v in sorted(gen_cls.items())},
    'mean_flips_32': float(gen_flips.mean()),
    'mean_flips_executed_rows0_9_of_9': float(np.mean(exec_flips)),
    'mean_flips_planned_rows10_31_of_22': float(np.mean(plan_flips)),
    'by_phase': {k: {'n': v['n'], 'mean_flips': v['flips'] / v['n'],
                     'p_any_flip': v['any'] / v['n']} for k, v in gen_phase.items()},
    'by_decile': [{'chunks': [i * 10, i * 10 + 9],
                   'mean_flips': float(gen_flips[i * 10:(i + 1) * 10].mean())} for i in range(6)],
    'corr_flips_vs_force': float(np.corrcoef(gen_flips, fmax_obs[[10 * k for k in range(60)]])[0, 1]),
    'corr_flips_vs_abs_dprev': float(np.corrcoef(gen_flips, [abs(p['dprev']) for p in per_chunk])[0, 1]),
    'per_chunk': per_chunk,
}

# ---------- matched comparison ----------
res['matched'] = {}
for b in ('static', 'micro', 'moving'):
    a = res['supervision']['by_phase'].get(b)
    c = res['generation_seed17']['by_phase'].get(b)
    res['matched'][b] = {'supervision': a, 'generation': c,
                         'ratio_mean_flips': (c['mean_flips'] / a['mean_flips'])
                         if a and c and a['mean_flips'] > 0 else None}

with open(OUT, 'w') as fh:
    json.dump(res, fh, indent=2, default=str)
print(json.dumps({k: res[k] for k in ('supervision', 'generation_seed17', 'matched')},
                 indent=1, default=str)[:6000])
print('WROTE', OUT)
os._exit(0)