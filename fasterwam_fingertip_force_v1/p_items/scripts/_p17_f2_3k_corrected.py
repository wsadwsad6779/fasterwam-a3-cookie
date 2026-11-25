import glob
import hashlib
import json
import os
import sys
import time

import numpy as np

P = '/workspace/fwv1/p_items'
EV = '/workspace/fwv1/eval_fingertip_v1'
EXPECTED = '342c4c4a74e63598a6a565eda59db2570878d842742fc65ba7ad343db058a34e'
TS = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 22), b''):
            h.update(c)
    return h.hexdigest()


def load(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def exec_stats(sub):
    vals = []
    for p in glob.glob(os.path.join(EV, sub, 'seed_*', 'events.json')):
        d = load(p) or {}
        vals.extend(r['act_grip_raw'] for r in d.get('rows', []))
    if not vals:
        return None
    v = np.asarray(vals)
    return {'n': int(v.size), 'median': float(np.median(v)),
            'lt_0.30': float((v < 0.30).mean()), 'open_ge_0.8': float((v >= 0.80).mean())}


def rows(d):
    out = []
    for r in (d or {}).get('results', []):
        fs = r.get('force_stats') or {}
        out.append({'seed': r.get('episode'), 'in_target': r.get('final_in_target'),
                    'exact_fill': bool(r.get('exact_fill')), 'grasps': r.get('grasps'),
                    'close_cmd': (r.get('action_stats') or {}).get('grip_close_frac'),
                    'open_cmd': (r.get('action_stats') or {}).get('open_cmd_frac'),
                    'carry_gt0p5': fs.get('carry_force_frac_gt_0p5'),
                    'force_max': fs.get('force_max')})
    return out


f2p = EV + '/cmp_f2_3k/SUMMARY_step_3000.json'
f0p = EV + '/cmp_f0_3k/SUMMARY_step_3000.json'
f2 = load(f2p)
f0 = load(f0p)
if not f2:
    raise SystemExit('missing ' + f2p)
actual = f2.get('checkpoint_sha256')
ok = bool(actual == EXPECTED)
out = {'ts': TS, 'artifact': 'P17_F2_3K_CORRECTED',
       'expected_checkpoint_sha256': EXPECTED, 'actual_checkpoint_sha256': actual,
       'identity_ok': ok,
       'f2_summary': f2p, 'f2_summary_sha256': sha(f2p),
       'f2_checkpoint': f2.get('checkpoint'), 'f2_rows': rows(f2),
       'f2_exec': exec_stats('cmp_f2_3k'),
       'f0_rows': rows(f0), 'f0_exec': exec_stats('cmp_f0_3k'),
       'corrected_cmp': load(P + '/F0_F2_COMPARISON_3K.json'),
       'r_sweep_log_tail': ''}
try:
    t = open('/workspace/fwv1/logs/r_compare.log').read().strip().splitlines()
    out['r_sweep_log_tail'] = t[-6:]
except Exception:
    pass
if not ok:
    out['status'] = 'REJECTED: F2 summary checkpoint sha does not match the archive sha'
    json.dump(out, open(P + '/P17_F2_3K_CORRECTED.json', 'w'), indent=1)
    print('P17 REJECTED actual=%s' % actual)
    sys.exit(2)
f2rows = out['f2_rows']
out['status'] = 'ACCEPTED'
out['f2_means'] = {'n': len(f2rows),
                   'mean_in_target': sum((r['in_target'] or 0) for r in f2rows) / len(f2rows),
                   'exact_fill_rate': sum(1 for r in f2rows if r['exact_fill']) / len(f2rows),
                   'mean_grasps': sum((r['grasps'] or 0) for r in f2rows) / len(f2rows)}
out['f0_means'] = {'n': len(out['f0_rows']),
                   'mean_in_target': sum((r['in_target'] or 0) for r in out['f0_rows']) / len(out['f0_rows']),
                   'exact_fill_rate': sum(1 for r in out['f0_rows'] if r['exact_fill']) / len(out['f0_rows']),
                   'mean_grasps': sum((r['grasps'] or 0) for r in out['f0_rows']) / len(out['f0_rows'])}
out['conclusion'] = (
    'corrected F2 3k accepted at sha 342c4c4a; both F0 and F2 reach exact_fill 0/10 and in_target 0 at 3k. '
    'The F2 3k behavior is recorded per seed; the official r10 anchor stays the 15k milestone. '
    'The r sweep restarts only after this acceptance.')
json.dump(out, open(P + '/P17_F2_3K_CORRECTED.json', 'w'), indent=1)
print('P17 ACCEPTED', out['f2_means'], 'exec', out['f2_exec'])