"""Offline analysis of gpu_mechanism_v6_20261118. CPU only, no model, no simulator."""
import json
import hashlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'gpu_mechanism_v6_20261118'
STATS = json.load(open(ROOT / 'dataset_stats_14d_RECONSTRUCTED.json'))
CALIB = json.load(open(ROOT.parent / 'step_040000' / 'force_calib_v1.json'))
LO = np.asarray(STATS['action']['default']['global_min'], dtype=np.float64)
HI = np.asarray(STATS['action']['default']['global_max'], dtype=np.float64)

MARKERS = {
    'NORMALIZER_IDENTITY': 'RECONSTRUCTED_DETERMINISTIC_NOT_ORIGINAL',
    'original_stats_sha256': 'UNKNOWN',
    'scope': 'offline predicted-plan comparison at recorded replan states; no control executed',
}

jobs = {}
for f in sorted(OUT.glob('*.json')):
    if f.name in ('summary.json', 'identity.json', 'predeclared_jobs.json'):
        continue
    try:
        d = json.loads(f.read_text())
    except Exception:
        continue
    if isinstance(d, dict) and 'name' in d and 'frame' in d:
        jobs[d['name']] = d

assert jobs, 'no job json found'


def raw(name):
    z = np.load(OUT / (name + '.npz'))
    return (z['raw32'].astype(np.float64), z['force_future_newton'].astype(np.float64),
            z['recorded_model_out'].astype(np.float64), z['recorded_adapter_out'].astype(np.float64))


def to_cmd(raw32):
    x = np.asarray(raw32, dtype=np.float64)
    den = (x + 1.0) / 2.0 * (HI - LO) + LO
    pm1 = den.copy()
    pm1[:, 6] = np.where(pm1[:, 6] >= 0.0, 1.0, -1.0)
    out = np.zeros((x.shape[0], 14))
    out[:, :3] = np.clip(pm1[:, :3] * 2.0, -1.0, 1.0)
    out[:, 3:6] = np.clip(pm1[:, 3:6] * 0.5 / 0.1, -1.0, 1.0)
    out[:, 6] = -np.sign(pm1[:, 6])
    return out


def plan_stats(raw32):
    cmd = to_cmd(raw32[:10])
    trans = np.abs(cmd[:, :3])
    return {
        'a3_trans_absmax_first10': float(trans.max()),
        'a3_trans_saturated_rows_first10': int((trans.max(1) >= 1.0 - 1e-9).sum()),
        'a3_gripper_cmd_first10': cmd[:, 6].tolist(),
        'raw_absmax_32': float(np.abs(raw32).max()),
        'raw_absmax_first10': float(np.abs(raw32[:10]).max()),
    }


def frame_of(name):
    return int(jobs[name]['frame'])


names_by_frame = {}
for n, d in jobs.items():
    names_by_frame.setdefault(int(d['frame']), {})[n.split('_f%d_' % int(d['frame']))[-1]] = n

frames = sorted(names_by_frame)
table = {}
for f in frames:
    entry = {}
    a0_name = 'replan_s%d_f%d_A0' % (17, f)
    a0_raw, a0_fut, rec_raw, rec_ad = raw(a0_name)
    entry['A0'] = {
        'plan': plan_stats(a0_raw),
        'vs_recorded': jobs[a0_name]['compare_recorded'],
        'future_absmax': float(np.abs(a0_fut).max()),
        'history_changed_entries': jobs[a0_name]['history_changed_entries'],
        'calib_check_translation_gripper': {
            'trans_max_abs_diff_vs_recorded_adapter': float(np.abs(to_cmd(a0_raw)[:10, :3] - rec_ad[:10, :3]).max()),
            'gripper_equal_rows': int((to_cmd(a0_raw)[:10, 6] == rec_ad[:10, 6]).sum()),
        },
    }
    for kind, nm in names_by_frame[f].items():
        if kind == 'A0':
            continue
        r, fut, _, _ = raw(nm)
        d = jobs[nm]
        delta = np.abs(r - a0_raw)
        entry[kind] = {
            'plan': plan_stats(r),
            'vs_a0': {
                'max_abs_delta': float(delta.max()),
                'first10_max_abs_delta': float(delta[:10].max()),
                'rows_changed_gt_1e-3': int((delta.max(1) > 1e-3).sum()),
                'dir_sign_agree_first10': [int((np.sign(r[:10, j]) == np.sign(a0_raw[:10, j])).sum())
                                           for j in range(6)],
                'gripper_sign_agree_first10': int((np.sign(r[:10, 6]) == np.sign(a0_raw[:10, 6])).sum()),
            },
            'future_absmax': float(np.abs(fut).max()),
            'future_max_abs_delta_vs_a0': float(np.abs(fut - a0_fut).max()),
            'history_changed_entries': d['history_changed_entries'],
            'proprio_norm_force_used': d['proprio_norm_used'][8:14],
        }
    table[f] = entry

repeat = {}
if 'replan_s17_f140_A0_repeat' in jobs:
    a, _, _, _ = raw('replan_s17_f140_A0')
    b, _, _, _ = raw('replan_s17_f140_A0_repeat')
    repeat = {'bit_equal': bool(np.array_equal(a, b)), 'max_abs_diff': float(np.abs(a - b).max())}

hooks = {}
for f in frames:
    a0 = jobs['replan_s17_f%d_A0' % f]['force_hooks']
    rows = {}
    for mod, rec in a0.items():
        row = {'A0': rec['max_abs']}
        for kind, nm in names_by_frame[f].items():
            if kind == 'A0':
                continue
            h = jobs[nm]['force_hooks'].get(mod)
            row[kind] = None if h is None else h['max_abs']
        rows[mod] = row
    hooks[f] = rows

summary = {
    **MARKERS,
    'jobs_analyzed': sorted(jobs),
    'a0_repeat_f140': repeat,
    'per_frame': table,
    'force_hook_stage_maxabs': hooks,
    'notes': [
        'raw32 is the model-normalised action; A3 commands are reconstructed with the identified min/max de-normaliser and only dims 0..2/6 are rotation-free.',
        'vs_recorded compares A0 with the chunk that was actually computed at that replan during the recorded episode.',
        'A sign disagreement is a plan difference, not a success or safety verdict.',
    ],
}
(OUT / 'analysis_v6.json').write_text(json.dumps(summary, indent=2, default=str))

lines = []
lines.append('frame  A0_absmax  rec_absmax  A0-vs-rec(max|d|, gripAgree/10)  '
             'clip_all(d,chg)  P1neutral(d,chg)  P1P2(d,chg)  clip_current(d,chg)  clip_history(d,chg)')
for f in frames:
    e = table[f]
    a0 = e['A0']
    rec = a0['vs_recorded']['vs_recorded_full32']['max_abs_diff'] if 'vs_recorded_full32' in a0['vs_recorded'] else None
    recmax = a0['vs_recorded']['raw_absmax_32']
    grip = a0['vs_recorded']['vs_recorded_first10']['gripper_sign_agree']
    cells = []
    for kind in ('P2clip_all', 'P1neutral', 'P1P2', 'P2clip_current', 'P2clip_history'):
        if kind in e:
            v = e[kind]['vs_a0']
            cells.append('%7.4f/%2d' % (v['max_abs_delta'], e[kind]['history_changed_entries']))
        else:
            cells.append('%12s' % '-')
    lines.append('%-5d %10.4f %10.4f %10.4f %2d/10 %s' % (
        f, a0['plan']['raw_absmax_32'], recmax, rec, grip, '  '.join(cells)))
(OUT / 'analysis_v6.txt').write_text('\n'.join(lines) + '\n')
print('\n'.join(lines))
print('repeat', repeat)
print('wrote', OUT / 'analysis_v6.json')
print('sha256', hashlib.sha256((OUT / 'analysis_v6.json').read_bytes()).hexdigest())
