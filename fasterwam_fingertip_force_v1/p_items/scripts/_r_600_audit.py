import json
import os
import time

import numpy as np

EV = '/workspace/fwv1/eval_fingertip_v1'
P = '/workspace/fwv1/p_items'
TS = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def load(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


out = {'ts': TS, 'artifact': 'R_SWEEP_600STEP_AUDIT',
       'issue': 'r16/r32 chunks cover 608 nominal steps (38*16 / 19*32) but the executor stops at 600; the last chunk is truncated',
       'official_metrics': 'in_target/exact/grasps from SUMMARY are computed at exactly 600 control steps (events rows)',
       'method': 'recompute executed gripper stats with take=min(r, 600-t) per chunk; compare with the naive g[:r] overcount',
       'runs': {}}
for r in (10, 16, 32):
    root = os.path.join(EV, 'diag15k_r%d_grid' % r)
    d = load(os.path.join(root, 'SUMMARY_step_diag15k_r%d.json' % r))
    if not d:
        out['runs']['r%d' % r] = {'missing': True}
        continue
    res = {int(x.get('episode')): x for x in d.get('results', [])}
    rows = []
    naive_all, corr_all = [], []
    for seed in range(7500, 7510):
        ch = (load(os.path.join(root, 'seed_%05d' % seed, 'chunks.json')) or {}).get('chunks', [])
        ev = load(os.path.join(root, 'seed_%05d' % seed, 'events.json')) or {}
        naive, corr = [], []
        for c in ch:
            g = [row[6] for row in c.get('denorm', []) if len(row) > 6]
            naive.extend(g[:r])
            take = min(r, 600 - int(c.get('t', 0)))
            corr.extend(g[:max(0, take)])
        x = res.get(seed, {})
        rows.append({'seed': seed, 'chunks': len(ch), 'naive_steps': len(naive), 'correct_steps': len(corr),
                     'events_rows': len(ev.get('rows', [])),
                     'exec_lt030_naive': float((np.asarray(naive) < 0.30).mean()) if naive else None,
                     'exec_lt030_correct': float((np.asarray(corr) < 0.30).mean()) if corr else None,
                     'exec_median_correct': float(np.median(corr)) if corr else None,
                     'in_target_strict': x.get('final_in_target'),
                     'in_target_user': x.get('final_in_target_relaxed'),
                     'exact_fill': bool(x.get('exact_fill')),
                     'last_chunk_t': int(ch[-1].get('t')) if ch else None,
                     'last_take': min(r, 600 - int(ch[-1].get('t', 0))) if ch else None})
        naive_all.extend(naive)
        corr_all.extend(corr)
    out['runs']['r%d' % r] = {
        'n_seeds': len(rows),
        'exec_lt030_naive': float((np.asarray(naive_all) < 0.30).mean()),
        'exec_lt030_correct': float((np.asarray(corr_all) < 0.30).mean()),
        'delta_pp': float(100 * ((np.asarray(corr_all) < 0.30).mean() - (np.asarray(naive_all) < 0.30).mean())),
        'exec_median_correct': float(np.median(corr_all)),
        'mean_in_target_user': float(np.mean([x['in_target_user'] or 0 for x in rows])),
        'mean_in_target_strict': float(np.mean([x['in_target_strict'] or 0 for x in rows])),
        'exact_fill_rate': float(np.mean([1 if x['exact_fill'] else 0 for x in rows])),
        'step600_vs_608_in_box_change': 'none: the rollout stops at t=600; the summary/events are 600 steps. Only the naive chunk-level <0.30 stats change.',
        'rows': rows}
    print('r=%d naive=%.5f correct=%.5f delta=%.3fpp' % (
        r, out['runs']['r%d' % r]['exec_lt030_naive'], out['runs']['r%d' % r]['exec_lt030_correct'],
        out['runs']['r%d' % r]['delta_pp']))
json.dump(out, open(P + '/R_SWEEP_600STEP_AUDIT.json', 'w'), indent=1)
print('WROTE', P + '/R_SWEEP_600STEP_AUDIT.json')