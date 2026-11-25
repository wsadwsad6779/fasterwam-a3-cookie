import csv
import json
import os

EV = '/workspace/fwv1/eval_fingertip_v1'
P = '/workspace/fwv1/p_items'
RUNS = [
    {'replan': 10, 'summary': EV + '/diag15k_r10_grid/SUMMARY_step_diag15k_r10.json',
     'root': EV + '/diag15k_r10_grid'},
    {'replan': 16, 'summary': EV + '/diag15k_r16_grid/SUMMARY_step_diag15k_r16.json',
     'root': EV + '/diag15k_r16_grid'},
    {'replan': 32, 'summary': EV + '/diag15k_r32_grid/SUMMARY_step_diag15k_r32.json',
     'root': EV + '/diag15k_r32_grid'},
]
OFFICIAL = EV + '/step_015000/SUMMARY_step_15000.json'


def load(path):
    if not path or not os.path.exists(path):
        return None
    try:
        return json.load(open(path))
    except Exception:
        return None


def chunk_stats(path, replan):
    d = load(path)
    if not d:
        return None
    vals = []
    inf = []
    for c in d.get('chunks', []):
        rows = c.get('denorm') or []
        g = [row[6] for row in rows if len(row) > 6]
        take = min(replan, 600 - int(c.get('t', 0)))
        vals.extend(g[:max(0, take)])
        if c.get('infer_ms') is not None:
            inf.append(float(c['infer_ms']))
    if not vals:
        return None
    s = sorted(vals)
    return {'exec_lt030': sum(1 for v in vals if v < 0.30) / len(vals),
            'exec_lt035': sum(1 for v in vals if v < 0.35) / len(vals),
            'exec_median': s[len(s) // 2], 'exec_n': len(vals),
            'chunk_infer_ms_mean': (sum(inf) / len(inf)) if inf else None}


out = {'worker': 'task7', 'study': 'R1_REPLAN_SWEEP',
       'checkpoint': '/workspace/fwv1/train_runs/t7_21k_v3run/step_015000.pt',
       'checkpoint_sha256': None,
       'scene_manifest': '/workspace/fwv1/p_items/EVAL_SCENE_MANIFEST_V3_1.json',
       'counting_rule': {
           'primary': 'user rule: footprint inside target box AND bottom edge on the box floor (within 4 mm); upright not required',
           'strict': 'sim cookies_in_target requires tilt <= 15deg; exact_fill unchanged',
           'note': 'all r rerun with the diagnostic script and the relaxed counter so the curve uses one rule'},
       'protocol': {'max_steps': 600, 'denoise': 5, 'action_horizon': 32,
                    'seeds': list(range(7500, 7510)), 'pad': 0.0005,
                    'eval_script': '/workspace/_t7_eval_diag.py',
                    'official_replan': 10,
                    'valid_r': '1..32 (effective_r = min(32, requested_r)); r>32 is an alias of r32'},
       'official_r10_strict': None,
       'runs': []}

off = load(OFFICIAL)
off_map = {}
if off:
    off_map = {r.get('episode'): r for r in off.get('results', [])}
    orows = [r.get('final_in_target') or 0 for r in off.get('results', [])]
    out['official_r10_strict'] = {
        'source': 'official milestone eval step_015000 (strict sim count, no relaxed counter)',
        'n_seeds': len(orows), 'mean_in_target': sum(orows) / len(orows) if orows else None,
        'exact_fill_rate': sum(1 for r in off.get('results', []) if r.get('exact_fill')) / max(1, len(orows)),
        'per_seed': [{'seed': r.get('episode'), 'in_target': r.get('final_in_target'),
                      'exact_fill': bool(r.get('exact_fill')), 'grasps': r.get('grasps')}
                     for r in off.get('results', [])]}

rows_csv = []
for run in RUNS:
    d = load(run['summary'])
    if not d:
        out['runs'].append({'replan': run['replan'], 'missing': run['summary']})
        continue
    out['checkpoint_sha256'] = out['checkpoint_sha256'] or d.get('checkpoint_sha256')
    rows = []
    for res in d.get('results', []):
        seed = res.get('episode')
        fs = res.get('force_stats') or {}
        row = {'seed': seed,
               'in_target_strict': res.get('final_in_target'),
               'in_target_user': res.get('final_in_target_relaxed'),
               'final_in_source': res.get('final_in_source'),
               'exact_fill': bool(res.get('exact_fill')),
               'grasps': res.get('grasps'),
               'close_cmd_frac': res.get('close_cmd_frac'),
               'carry_gt0p5': fs.get('carry_force_frac_gt_0p5'),
               'force_max': fs.get('force_max'),
               'elapsed_s': res.get('elapsed_s'),
               'replan_calls': res.get('replan_calls'),
               'inference_ms_mean': res.get('inference_ms_mean'),
               'strict_reimpl_mismatch_steps': res.get('strict_reimpl_mismatch_steps')}
        st = chunk_stats(run['root'] + '/seed_%05d/chunks.json' % seed, run['replan'])
        if st:
            row.update(st)
        rows.append(row)
        rows_csv.append(dict({'r': run['replan']}, **row))
    n = len(rows)
    strict = [r['in_target_strict'] or 0 for r in rows]
    user = [(r['in_target_user'] if r['in_target_user'] is not None else 0) for r in rows]
    has_user = [r for r in rows if r['in_target_user'] is not None]
    out['runs'].append({
        'replan': run['replan'], 'n_seeds': n,
        'label': 'relaxed-r10 (diagnostic)' if run['replan'] == 10 else 'diag-r%d' % run['replan'],
        'official_anchor': run['replan'] == 10,
        'official_strict_match': None,
        'effective_r': min(32, run['replan']),
        'mean_in_target_user': sum(user) / n if n else None,
        'mean_in_target_strict': sum(strict) / n if n else None,
        'max_in_target_user': max(user) if n else None,
        'placement_fraction_user': sum(user) / (10.0 * n) if n else None,
        'exact_fill_rate': sum(1 for r in rows if r['exact_fill']) / n if n else None,
        'mean_grasps': sum(r['grasps'] or 0 for r in rows) / n if n else None,
        'mean_elapsed_s': (sum(r['elapsed_s'] for r in rows if r['elapsed_s'] is not None)
                           / max(1, sum(1 for r in rows if r['elapsed_s'] is not None))),
        'mean_replan_calls': (sum(r['replan_calls'] for r in rows if r['replan_calls'] is not None)
                              / max(1, sum(1 for r in rows if r['replan_calls'] is not None))),
        'mean_inference_ms': (sum(r['inference_ms_mean'] for r in rows if r['inference_ms_mean'] is not None)
                              / max(1, sum(1 for r in rows if r['inference_ms_mean'] is not None))),
        'mean_exec_lt030': (sum(r['exec_lt030'] for r in rows if 'exec_lt030' in r)
                            / max(1, sum(1 for r in rows if 'exec_lt030' in r))),
        'n_exec_stats': sum(1 for r in rows if 'exec_lt030' in r),
        'n_user_counter': len(has_user),
        'rows': rows})
    if run['replan'] == 10 and off_map:
        per = []
        for row in rows:
            o = off_map.get(row['seed'])
            if not o:
                continue
            per.append({'seed': row['seed'],
                        'in_target_equal': row['in_target_strict'] == o.get('final_in_target'),
                        'exact_fill_equal': bool(row['exact_fill']) == bool(o.get('exact_fill')),
                        'grasps_equal': (row['grasps'] or 0) == (o.get('grasps') or 0)})
        ok = bool(per) and all(x['in_target_equal'] and x['exact_fill_equal'] and x['grasps_equal']
                               for x in per)
        out['runs'][-1]['official_strict_match'] = {
            'all': ok, 'per_seed': per, 'basis': 'in_target + exact_fill + grasps'}
    rr = out['runs'][-1]
    print('r=%d user=%s strict=%s exact=%s n_user=%d' % (
        run['replan'], rr['mean_in_target_user'], rr['mean_in_target_strict'],
        rr['exact_fill_rate'], rr['n_user_counter']))

json.dump(out, open(P + '/R_COMPARE_15K.json', 'w'), indent=1)
json.dump({'study': 'R1_REPLAN_SWEEP', 'checkpoint_sha256': out['checkpoint_sha256'],
           'counting_rule': out['counting_rule'],
           'runs': [{k: v for k, v in r.items() if k != 'rows'} for r in out['runs']]},
          open(P + '/R_SWEEP_SUMMARY.json', 'w'), indent=1)
cols = ['r', 'seed', 'in_target_user', 'in_target_strict', 'exact_fill', 'grasps',
        'carry_gt0p5', 'force_max', 'close_cmd_frac', 'exec_lt030', 'exec_lt035',
        'exec_median', 'elapsed_s', 'replan_calls', 'inference_ms_mean', 'strict_reimpl_mismatch_steps']
with open(P + '/R_SWEEP_RAW_RESULTS.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
    w.writeheader()
    for r in rows_csv:
        w.writerow(r)
json.dump({'done': True, 'runs': len(out['runs'])}, open(P + '/R_COMPARE_DONE.json', 'w'), indent=1)
print('WROTE', P + '/R_COMPARE_15K.json', P + '/R_SWEEP_SUMMARY.json', P + '/R_SWEEP_RAW_RESULTS.csv')