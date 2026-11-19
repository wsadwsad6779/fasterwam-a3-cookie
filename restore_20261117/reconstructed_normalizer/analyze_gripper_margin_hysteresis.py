"""Is the model's gripper chatter driven by a shallow decision margin, and does hysteresis fix it?

CPU only, reads the recorded seed17 chunks. No model, no simulator.
"""
import json

import numpy as np

BASE = '/workspace/eval_restore_20261117/reconstructed_normalizer'
FORE = BASE + '/forensics_full_wide_out'
OUT = BASE + '/GRIPPER_MARGIN_HYSTERESIS.json'
H = 32

cz = np.load(FORE + '/seed17_original_chunks.npz')
chunks = np.stack([np.asarray(cz['chunk_%03d_model_out' % k], np.float64) for k in range(60)])
vals = chunks[:, :, 6]
sg = np.where(vals >= 0, 1, -1)


def flips(s):
    return np.flatnonzero(np.diff(s) != 0)


def hysteresis(v, h, prev=1):
    out = np.empty_like(v)
    s = prev
    for t, x in enumerate(v):
        if abs(x) >= h:
            s = 1 if x >= 0 else -1
        out[t] = s
    return out


rows = []
for k in range(60):
    v = vals[k]
    s = sg[k]
    f = flips(s)
    # decision margin at each flip: min |v| over the two rows that disagree
    margins = [float(min(abs(v[t - 1]), abs(v[t]))) for t in f]
    rows.append({'chunk': k, 'frame': 10 * k, 'n_flips': int(len(f)),
                 'flip_rows': f.tolist(), 'flip_margins': margins,
                 'min_abs_exec': float(np.abs(v[:10]).min()),
                 'mean_abs_all': float(np.abs(v).mean()),
                 'sign10': ''.join('+' if x > 0 else '-' for x in s[:10]),
                 'flips_exec': int((np.diff(s[:10]) != 0).sum())})

allmarg = np.concatenate([r['flip_margins'] for r in rows if r['flip_margins']] or [np.zeros(0)])
chatter = [r for r in rows if r['n_flips'] >= 6]
clean = [r for r in rows if r['n_flips'] <= 1]

res = {'NORMALIZER_IDENTITY': 'RECONSTRUCTED_DETERMINISTIC_NOT_ORIGINAL',
       'original_stats_sha256': 'UNKNOWN',
       'flip_decision_margin': {
           'n_flips': int(allmarg.size),
           'p50': float(np.percentile(allmarg, 50)) if allmarg.size else None,
           'p90': float(np.percentile(allmarg, 90)) if allmarg.size else None,
           'frac_lt_0_05': float((allmarg < 0.05).mean()) if allmarg.size else None,
           'frac_lt_0_2': float((allmarg < 0.2).mean()) if allmarg.size else None,
           'frac_gt_0_8': float((allmarg > 0.8).mean()) if allmarg.size else None},
       'chatter_chunks': [{'chunk': r['chunk'], 'frame': r['frame'], 'flips': r['n_flips'],
                           'sign10': r['sign10'], 'min_abs_exec': r['min_abs_exec']} for r in chatter],
       'clean_chunks': len(clean),
       'per_chunk': rows}

# hysteresis sweep
sweep = {}
for h in (0.0, 0.1, 0.2, 0.5, 0.8, 0.95):
    tot_flips = 0
    tot_flips_exec = 0
    changed_intent = 0
    delayed = []
    for k in range(60):
        v = vals[k]
        s0 = sg[k]
        s1 = hysteresis(v, h, prev=1)
        tot_flips += int((np.diff(s1) != 0).sum())
        tot_flips_exec += int((np.diff(s1[:10]) != 0).sum())
        maj0 = 1 if (s0[:10] > 0).sum() >= 5 else -1
        maj1 = 1 if (s1[:10] > 0).sum() >= 5 else -1
        changed_intent += int(maj0 != maj1)
        f0 = flips(s0)
        f1 = flips(s1)
        if len(f0) == len(f1):
            delayed += (np.abs(f1 - f0)).tolist() if len(f0) else []
        else:
            delayed += ([999] * len(f1))
    sweep[str(h)] = {'total_flips_32': tot_flips,
                     'mean_flips_32': tot_flips / 60.0,
                     'mean_flips_exec_10': tot_flips_exec / 60.0,
                     'executed_majority_intent_changes': changed_intent,
                     'transition_shift_rows_median': float(np.median(delayed)) if delayed else None}
res['hysteresis_sweep'] = sweep

# comparison anchors
res['anchors'] = {
    'supervision_mean_flips_32': 1.2135,
    'supervision_mean_flips_first10_of_9pairs': 0.3909,
    'supervision_p_constant_32': 0.5183,
    'model_mean_flips_32_raw': float(np.mean([r['n_flips'] for r in rows])),
    'model_mean_flips_first10_raw': float(np.mean([r['flips_exec'] for r in rows])),
}

with open(OUT, 'w') as fh:
    json.dump(res, fh, indent=2, default=str)
print(json.dumps({k: res[k] for k in ('flip_decision_margin', 'chatter_chunks', 'hysteresis_sweep',
                                      'anchors')}, indent=1, default=str)[:4500])
print('WROTE', OUT)