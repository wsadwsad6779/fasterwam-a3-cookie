import glob
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


def chunk_exec(root, seed, r):
    d = load(os.path.join(root, 'seed_%05d', 'chunks.json' % seed))
    if not d:
        return None
    vals = []
    for c in d.get('chunks', []):
        g = [row[6] for row in c.get('denorm', []) if len(row) > 6]
        vals.extend(g[:r])
    if not vals:
        return None
    v = np.asarray(vals)
    return {'exec_lt_0.30': float((v < 0.30).mean()), 'exec_median': float(np.median(v))}


written = []
for r in (10, 16, 32):
    sub = 'diag15k_r%d_grid' % r
    root = os.path.join(EV, sub)
    sp = os.path.join(root, 'SUMMARY_step_diag15k_r%d.json' % r)
    d = load(sp)
    if not d:
        continue
    rows = []
    for res in d.get('results', []):
        seed = res.get('episode')
        row = {'seed': seed, 'in_target_user': res.get('final_in_target_relaxed'),
               'in_target_strict': res.get('final_in_target'), 'exact_fill': bool(res.get('exact_fill')),
               'grasps': res.get('grasps'),
               'open_cmd': (res.get('action_stats') or {}).get('open_cmd_frac'),
               'close_cmd': (res.get('action_stats') or {}).get('grip_close_frac')}
        st = chunk_exec(root, seed, r)
        if st:
            row.update(st)
        rows.append(row)
    n = len(rows)
    strict = [x['in_target_strict'] or 0 for x in rows]
    user = [(x['in_target_user'] if x['in_target_user'] is not None else 0) for x in rows]
    out = {'ts': TS, 'replan': r, 'label': 'official r10' if r == 10 else 'diagnostic-only r%d' % r,
           'official_anchor': r == 10, 'n_seeds': n,
           'mean_in_target_user': sum(user) / n if n else None,
           'mean_in_target_strict': sum(strict) / n if n else None,
           'exact_fill_rate': sum(1 for x in rows if x['exact_fill']) / n if n else None,
           'mean_grasps': sum(x['grasps'] or 0 for x in rows) / n if n else None,
           'mean_exec_lt_030': (sum(x['exec_lt_0.30'] for x in rows if 'exec_lt_0.30' in x)
                                / max(1, sum(1 for x in rows if 'exec_lt_0.30' in x))),
           'rows': rows, 'source_summary': sp}
    json.dump(out, open(os.path.join(P, 'R_COMPARE_PARTIAL_r%d.json' % r), 'w'), indent=1)
    written.append('r%d n=%d user=%s strict=%s' % (r, n, out['mean_in_target_user'], out['mean_in_target_strict']))
print(TS, 'partials:', written)