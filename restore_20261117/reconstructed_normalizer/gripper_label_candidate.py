"""Candidate gripper supervision from the expert's own joint command.

Original label: sign(opening[t+1]-opening[t]) with 1e-4 deadband, hold-previous (noisy).
Candidate: the expert's commanded opening joint_cmd[t] is a target position; compare it with the
measured opening and emit a hysteretic sign with a minimum hold:

    err = joint_cmd - opening
    err < -EPS  -> close (+1)          (target below measured: the gripper is asked to close)
    err > +EPS  -> open  (-1)
    |err| <= EPS -> hold previous
    minimum run of MIN_HOLD frames enforced

Reported per episode: flip count, run-length stats, agreement with the original label, number of
close runs >= 10 frames, number of releases, and agreement on the large-motion frames
(|d opening| > 0.05) where the original label is physically meaningful.

CPU only.
"""
import json

import numpy as np

A = '/workspace/eval_restore_20261117/archive_evidence/data/a3_fw/arrays.npz'
OUT = '/workspace/eval_restore_20261117/reconstructed_normalizer/GRIPPER_LABEL_CANDIDATE.json'
EPS = 0.02
MIN_HOLD = 3

z = np.load(A, allow_pickle=True)
act = np.asarray(z['action'], np.float64)
op = np.asarray(z['opening'], np.float64)
jc = np.asarray(z['joint_cmd'], np.float64)
es = np.asarray(z['ep_start'], np.int64)
el = np.asarray(z['ep_len'], np.int64)


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


rows = []
big_agree = []
for s, L in zip(es, el):
    s, L = int(s), int(L)
    o = op[s:s + L]
    c = jc[s:s + L]
    lab0 = act[s:s + L, 6]
    err = c - o
    lab1 = np.empty(L)
    prev = 1.0
    for t in range(L):
        if err[t] < -EPS:
            prev = 1.0
        elif err[t] > EPS:
            prev = -1.0
        lab1[t] = prev
    # enforce minimum hold: kill runs shorter than MIN_HOLD by merging
    for _ in range(3):
        rr = runs(lab1 > 0)
        rr += runs(lab1 <= 0)
        for (a, b, l) in rr:
            if l < MIN_HOLD:
                fill = 1.0 if (a > 0 and lab1[a - 1] > 0) else -1.0
                lab1[a:b + 1] = fill
    f0 = int((np.diff(lab0) != 0).sum())
    f1 = int((np.diff(lab1) != 0).sum())
    big = np.abs(np.diff(o)) > 0.05
    agree_big = float((np.sign(lab1[:-1])[big] == np.sign(lab0[:-1])[big]).mean()) if big.any() else None
    if agree_big is not None:
        big_agree.append(agree_big)
    close_runs = [l for (a, b, l) in runs(lab1 > 0) if l >= 10]
    rel = 0
    for (a, b, l) in runs(lab1 > 0):
        if l >= 10 and b + 1 < L and lab1[b + 1] < 0:
            rel += 1
    rows.append({'ep': s, 'len': L, 'flips_orig': f0, 'flips_cand': f1,
                 'agreement': float((np.sign(lab0) == np.sign(lab1)).mean()),
                 'agree_big_motion': agree_big, 'close_runs_ge10': len(close_runs),
                 'releases': rel, 'joint_cmd_min': float(c.min()), 'joint_cmd_max': float(c.max()),
                 'opening_min': float(o.min()), 'opening_max': float(o.max()),
                 'corr_cmd_opening': float(np.corrcoef(c, o)[0, 1])})

fo = np.array([r['flips_orig'] for r in rows], float)
fc = np.array([r['flips_cand'] for r in rows], float)
res = {'NORMALIZER_IDENTITY': 'RECONSTRUCTED_DETERMINISTIC_NOT_ORIGINAL',
       'TRAINING_DATA_IDENTITY': 'CANDIDATE_RECONSTRUCTED_TRAINING_DATA',
       'params': {'EPS': EPS, 'MIN_HOLD': MIN_HOLD},
       'flips_orig_mean': float(fo.mean()), 'flips_cand_mean': float(fc.mean()),
       'flips_ratio': float(fc.mean() / fo.mean()),
       'agreement_mean': float(np.mean([r['agreement'] for r in rows])),
       'agree_big_motion_mean': float(np.mean(big_agree)) if big_agree else None,
       'close_runs_ge10_orig_mean': float(np.mean([len([l for (a, b, l) in runs(act[int(r['ep']):int(r['ep']) + r['len'], 6] > 0) if l >= 10]) for r in rows])),
       'close_runs_ge10_cand_mean': float(np.mean([r['close_runs_ge10'] for r in rows])),
       'releases_cand_mean': float(np.mean([r['releases'] for r in rows])),
       'corr_cmd_opening_mean': float(np.mean([r['corr_cmd_opening'] for r in rows])),
       'per_episode': rows}
with open(OUT, 'w') as fh:
    json.dump(res, fh, indent=2, default=str)
print(json.dumps({k: v for k, v in res.items() if k != 'per_episode'}, indent=1))
print('WROTE', OUT)