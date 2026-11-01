import sys, json, hashlib
import numpy as np
from pathlib import Path
sys.path.insert(0, '/workspace/_r5_stage')
import replay_forces as R
ANOMALY_PAD_N = 1.6
HARD_CAP_N = 2.0
NOTE = 'segments closes a clamp only at an IDLE phase, so the recovery replan collapses the whole episode into one clamp with max_cookies_at_grip=5; driver accept is therefore always False. This file re-scores on the slice after the anomaly frame.'
args = [a for a in sys.argv[1:] if not a.startswith('--')]
write = '--write' in sys.argv
for base in args:
    d = Path(base)
    rows = json.loads((d / 'summary.json').read_text())
    tally = {'dir': str(d), 'n': 0, 'abnormal': 0, 'recovery_clean': 0, 'grasp_accept': 0, 'strict_accept': 0}
    out = []
    for row in rows:
        seed = int(row['seed'])
        p = d / ('expert_wrongclosure_%d.npz' % seed)
        if not p.exists():
            continue
        z = np.load(p)
        ph = [R.PHASE_NAMES[int(x)] for x in z['phase']]
        pad = np.asarray(z['pad_force_inner']); xyz = np.asarray(z['cookie_xyz'])
        eef = np.asarray(z['eef_xyz']); op = np.asarray(z['opening'])
        z0 = np.asarray(z['cookie_xyz0'])
        log = row['retry_log']
        rec = {'seed': seed, 'file': p.name, 'noise_m': row['noise_m'], 'recover_variant': row['recover_variant'],
               'steps': row['steps'], 'stop': row['stop'], 'driver_accept_field': row['accept'], 'cut': None,
               'abnormal': False, 'recovery_clean': False, 'grasp_accept': False, 'strict_accept': False,
               'wrong_closure_max_at_grip': None, 'wrong_closure_peak_pad_n': None, 'recovery_clamp': None}
        if log:
            rec['cut'] = int(log[0]['at_step'])
            cut = rec['cut']
            rec['wrong_closure_max_at_grip'] = int(log[0].get('max_at_grip', -1))
            rec['wrong_closure_peak_pad_n'] = round(float(np.max(pad[:cut + 1])), 3)
            rec['abnormal'] = bool(log[0]['anomaly_kind'] == 'wrong_closure'
                                   and rec['wrong_closure_max_at_grip'] >= 2
                                   and rec['wrong_closure_peak_pad_n'] <= ANOMALY_PAD_N)
            seg = pad[cut:]
            rec['recovery_clean'] = bool(int(np.sum(seg > HARD_CAP_N)) == 0 and float(seg.max()) <= ANOMALY_PAD_N)
            sl = ph[cut:]
            segs = R.segments(np.asarray(sl))
            if segs:
                a, b = segs[-1]
                st = R.clamp_stats(a, b, sl, pad[cut:], xyz[cut:], eef[cut:], z0, op[cut:], None)
                keys = ('phases', 'closure_peak_pad_n', 'carry_peak_pad_n', 'max_cookies_at_grip',
                        'carried_lift_mm', 'carried_index', 'placed_in_bin_xy', 'placed_upright', 'accept')
                rec['recovery_clamp'] = {k: st[k] for k in keys}
                rec['grasp_accept'] = bool(st['accept'])
        rec['strict_accept'] = bool(rec['abnormal'] and rec['recovery_clean'] and rec['grasp_accept'])
        tally['n'] += 1
        for k in ('abnormal', 'recovery_clean', 'grasp_accept', 'strict_accept'):
            tally[k] += int(rec[k])
        out.append(rec)
        small = ('seed', 'cut', 'abnormal', 'recovery_clean', 'grasp_accept', 'strict_accept', 'driver_accept_field')
        print(json.dumps({k: rec[k] for k in small}))
    print('TALLY', json.dumps(tally))
    if write:
        target = d / 'adjudication_wc.json'
        if target.exists():
            print('REFUSING to overwrite', str(target))
        else:
            blob = json.dumps({'note': NOTE, 'thresholds': {'anomaly_pad_n': ANOMALY_PAD_N,
                                'hard_cap_n': HARD_CAP_N}, 'tally': tally, 'episodes': out}, indent=2)
            target.write_text(blob, encoding='utf-8')
            print('WROTE', str(target), 'bytes', len(blob), 'sha', hashlib.sha256(blob.encode()).hexdigest()[:16])
