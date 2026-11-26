#!/usr/bin/env python3
"""Collect F0 r=10/16/32 diag results (strict/relaxed/ever-in-box) + pair with F2 r-sweep."""
import json
import os
import numpy as np

EV = '/workspace/fwv1/eval_fingertip_v1'
P = '/workspace/fwv1/p_items'
OUT = os.path.join(P, 'F0_R_SWEEP_RESULTS.json')
CSV = os.path.join(P, 'F0_R_SWEEP_RAW.csv')
RS = [10, 16, 32]
SEEDS = list(range(7500, 7510))


def load_f0(r):
    p = os.path.join(EV, 'f0_15k_r%d' % r, 'SUMMARY_step_f0r%d.json' % r)
    if not os.path.exists(p):
        return {'missing': p}
    d = json.load(open(p))
    per = {}
    ever = {}
    for res in d.get('results', []):
        s = res['episode']
        per[s] = {'strict': res.get('final_in_target'), 'relaxed': res.get('final_in_target_relaxed'),
                  'exact_fill': res.get('exact_fill'), 'grasps': res.get('grasps'),
                  'close_cmd_frac': res.get('close_cmd_frac'), 'safety': res.get('safety_reason')}
        ever[s] = 0
        tz = os.path.join(EV, 'f0_15k_r%d' % r, 'seed_%05d' % s, 'trajectory.npz')
        try:
            z = np.load(tz)
            if 'in_target_relaxed' in z:
                ever[s] = int(np.max(z['in_target_relaxed']))
        except Exception:
            pass
    def mean(key):
        vals = [per[s][key] for s in per if per[s][key] is not None]
        return float(np.mean(vals)) if vals else None
    return {'n': len(per),
            'mean_strict': mean('strict'), 'mean_relaxed': mean('relaxed'),
            'exact_fill': int(sum(1 for s in per if per[s]['exact_fill'])),
            'mean_grasps': mean('grasps'),
            'ever_in_box_mean': (float(np.mean(list(ever.values()))) if ever else None),
            'per_seed': {str(s): dict(per[s], ever_in_box=ever[s]) for s in per}}


def load_f2(r):
    p = os.path.join(P, 'R_COMPARE_PARTIAL_r%d.json' % r)
    if not os.path.exists(p):
        return {'missing': p}
    d = json.load(open(p))
    per = {}
    for row in d.get('rows', []):
        per[row['seed']] = {'strict': row.get('in_target_strict'), 'relaxed': row.get('in_target_user'),
                            'grasps': row.get('grasps'), 'exact_fill': row.get('exact_fill')}
    return {'n': len(per), 'mean_strict': d.get('mean_in_target_strict'),
            'mean_relaxed': d.get('mean_in_target_user'),
            'exact_fill': int(sum(1 for s in per if per[s]['exact_fill'])),
            'mean_grasps': d.get('mean_grasps'), 'per_seed': {str(s): per[s] for s in per}}


f0 = {r: load_f0(r) for r in RS}
f2 = {r: load_f2(r) for r in RS}
paired = {}
for s in SEEDS:
    paired[str(s)] = {}
    for r in RS:
        a = f0[r].get('per_seed', {}).get(str(s))
        b = f2[r].get('per_seed', {}).get(str(s))
        paired[str(s)][str(r)] = {'f0_relaxed': None if not a else a['relaxed'],
                                  'f2_relaxed': None if not b else b['relaxed'],
                                  'f0_strict': None if not a else a['strict'],
                                  'f2_strict': None if not b else b['strict']}
out = {'artifact': 'F0_R_SWEEP_RESULTS', 'ts': __import__('time').strftime('%Y-%m-%dT%H:%M:%SZ', __import__('time').gmtime()),
       'checkpoint_f0': '/workspace/fwv1/f0_v3_baseline/step_015000.pt',
       'checkpoint_f0_sha256': '8ef072416d6f25ac1a1ea91594a829200b467be906549e07efdeed65a9442723',
       'protocol': 'H=32, K=5, V3.1 frozen, 600 steps, r=min(r,32); SAME diag evaluator v2 for all r; seeds 7500-7509',
       'F0': f0, 'F2': f2, 'paired_by_seed': paired,
       'note': 'ever_in_box = max over episode of in_target_relaxed from trajectory.npz; strict/relaxed separate; r>32 aliases r32'}
json.dump(out, open(OUT, 'w'), indent=1)
with open(CSV, 'w') as f:
    f.write('model,r,seed,strict,relaxed,grasps,exact_fill,ever_in_box\n')
    for r in RS:
        for s in SEEDS:
            a = f0[r].get('per_seed', {}).get(str(s))
            if a:
                f.write('F0,%d,%d,%s,%s,%s,%s,%s\n' % (r, s, a['strict'], a['relaxed'], a['grasps'], a['exact_fill'], a['ever_in_box']))
    for r in RS:
        for s in SEEDS:
            b = f2[r].get('per_seed', {}).get(str(s))
            if b:
                f.write('F2,%d,%d,%s,%s,%s,%s,\n' % (r, s, b['strict'], b['relaxed'], b['grasps'], b['exact_fill']))
print(json.dumps({'F0': {r: {k: f0[r].get(k) for k in ('n', 'mean_strict', 'mean_relaxed', 'exact_fill', 'mean_grasps', 'ever_in_box_mean')} for r in RS},
                  'F2': {r: {k: f2[r].get(k) for k in ('n', 'mean_strict', 'mean_relaxed', 'exact_fill', 'mean_grasps')} for r in RS}}))
