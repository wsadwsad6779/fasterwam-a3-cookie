#!/usr/bin/env python3
"""Collect the H sweep (32/48/64) closed-loop in-box results into one ledger."""
import json
import os

ROOT = '/workspace/fwv1/eval_fingertip_v1'
OUT = '/workspace/fwv1/p_items/H_SWEEP_RESULTS.json'
res = {}
for H in (32, 48, 64):
    p = os.path.join(ROOT, 'h_eval_h%d' % H, 'SUMMARY_step_h%d.json' % H)
    if os.path.exists(p):
        d = json.load(open(p))
        res[str(H)] = {
            'summary': p,
            'avg_in_target_relaxed': d.get('avg_in_target_relaxed'),
            'avg_in_target_strict': d.get('avg_in_target'),
            'exact_fill_rate': d.get('exact_fill_rate'),
            'n_seeds': len(d.get('results', [])),
            'effective_r': (d.get('protocol') or {}).get('replan'),
            'checkpoint_sha256': d.get('checkpoint_sha256'),
        }
    else:
        res[str(H)] = {'missing': p}
out = {
    'sweep': 'H closed-loop in-box (variable H, r=H), F2 15k',
    'model': '/workspace/fwv1/train_runs/t7_21k_v3run/step_015000.pt',
    'note': 'H>32 positions untrained => exploratory OOD; report observed in-box only, not capability',
    'results': res,
}
json.dump(out, open(OUT, 'w'), indent=1)
print(json.dumps(res), flush=True)
