"""Offline analysis of closed_loop_p2clamp_20261118_v2 (B0 vs P2-clamp). CPU only."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'closed_loop_p2clamp_20261118_v2'
CAL = json.load(open(ROOT.parent / 'step_040000' / 'force_calib_v1.json'))
CHANS = [int(c) for c in CAL['channels']]
NAMES = list(CAL['channel_names'])

MARK = {'NORMALIZER_IDENTITY': 'RECONSTRUCTED_DETERMINISTIC_NOT_ORIGINAL',
        'original_stats_sha256': 'UNKNOWN'}


def load(tag):
    z = np.load(OUT / (tag + '.npz'), allow_pickle=False)
    d = json.loads((OUT / (tag + '.json')).read_text())
    eef = np.asarray(z['step_eef_pose'], dtype=np.float64)
    force18 = np.asarray(z['step_force18'], dtype=np.float64)
    f = force18[:, CHANS]
    n = len(f)
    cmd = np.zeros((n, 3))
    grip = np.zeros(n)
    plan_absmax = []
    plan_first10 = []
    for k in range(n // 10):
        ad = np.asarray(z['chunk_%03d_adapter_out' % k], dtype=np.float64).reshape(-1, 14)
        mo = np.asarray(z['chunk_%03d_model_out' % k], dtype=np.float64).reshape(-1, 7)
        plan_absmax.append(float(np.abs(mo).max()))
        plan_first10.append(float(np.abs(mo[:10]).max()))
        m = min(10, n - 10 * k)
        cmd[10 * k:10 * k + m] = ad[:m, :3] * 0.025
        grip[10 * k:10 * k + m] = ad[:m, 6]
    disp = np.zeros_like(cmd)
    disp[:-1] = eef[1:, :3] - eef[:-1, :3]
    csz = np.linalg.norm(cmd, axis=1)
    dsz = np.linalg.norm(disp, axis=1)
    track = np.where(csz > 1e-6, dsz / np.maximum(csz, 1e-9), np.nan)
    return {'z': z, 'json': d, 'eef': eef, 'force': f, 'fmax': np.abs(f).max(1),
            'cmd': cmd, 'disp': disp, 'track': track, 'grip': grip,
            'plan_absmax': np.asarray(plan_absmax), 'plan_first10': np.asarray(plan_first10)}


def period(a, lo, hi):
    sl = slice(lo, min(hi, len(a)))
    return a[sl]


def summarize(tag, a):
    fmax, track, cmd = a['fmax'], a['track'], a['cmd']
    csz = np.linalg.norm(cmd, axis=1)
    jam = (csz > 0.02) & np.isfinite(track) & (track < 0.3)
    rec = a['json']['record']
    out = {
        'arm': tag,
        'frames': int(len(fmax)),
        'success': rec['success'], 'cookies_in_target': rec['cookies_in_target'],
        'force_absmax': float(fmax.max()),
        'force_absmax_by_period': {
            '0-100': float(fmax[:100].max()) if len(fmax) > 100 else None,
            '100-200': float(fmax[100:200].max()) if len(fmax) > 200 else None,
            '200-300': float(fmax[200:300].max()) if len(fmax) > 300 else None,
            '300-400': float(fmax[300:400].max()) if len(fmax) > 400 else None,
            '400-500': float(fmax[400:500].max()) if len(fmax) > 500 else None,
            '500-600': float(fmax[500:600].max()) if len(fmax) > 600 else None,
        },
        'frames_gt_14_3N': int((fmax > 14.3).sum()),
        'frames_gt_50N': int((fmax > 50.0).sum()),
        'frames_gt_100N': int((fmax > 100.0).sum()),
        'plan_absmax_max': float(a['plan_absmax'].max()),
        'plan_absmax_by_chunk10': a['plan_absmax'].reshape(6, 10).max(1).tolist(),
        'jam_steps_cmd_gt20mm_track_lt0.3': int(jam.sum()),
        'first_jam_frame': int(np.argmax(jam)) if jam.any() else None,
        'jam_longest_run': int(max((len(r) for r in ''.join('1' if x else '0' for x in jam).split('0')), default=0)),
        'mean_track_free_steps_cmd_gt20mm': float(np.nanmean(track[csz > 0.02])) if (csz > 0.02).any() else None,
        'final_eef': a['eef'][-1, :3].tolist(),
        'gripper_closed_fraction': float((a['grip'] < 0).mean()),
    }
    return out


if not (OUT / 'B0.npz').exists() or not (OUT / 'CL.npz').exists():
    raise SystemExit('missing B0.npz or CL.npz in %s' % OUT)

b0 = load('B0')
cl = load('CL')
sb0, scl = summarize('B0', b0), summarize('CL', cl)

first_div = None
n = min(len(b0['force']), len(cl['force']))
for t in range(n):
    if not np.array_equal(b0['force'][t], cl['force'][t]):
        first_div = t
        break

# frame-level trace around the jam onset in both arms
rows = []
for t in range(0, n, 5):
    rows.append({
        'frame': t,
        'B0_fmax': float(b0['fmax'][t]), 'CL_fmax': float(cl['fmax'][t]),
        'B0_cmd_mm': float(np.linalg.norm(b0['cmd'][t]) * 1000),
        'CL_cmd_mm': float(np.linalg.norm(cl['cmd'][t]) * 1000),
        'B0_disp_mm': float(np.linalg.norm(b0['disp'][t]) * 1000),
        'CL_disp_mm': float(np.linalg.norm(cl['disp'][t]) * 1000),
        'B0_track': None if not np.isfinite(b0['track'][t]) else float(b0['track'][t]),
        'CL_track': None if not np.isfinite(cl['track'][t]) else float(cl['track'][t]),
        'B0_grip': float(b0['grip'][t]), 'CL_grip': float(cl['grip'][t]),
    })

chunk_tbl = []
for k in range(len(b0['plan_absmax'])):
    sl = slice(10 * k, 10 * k + 10)
    chunk_tbl.append({
        'chunk': k, 'frame': 10 * k,
        'B0_plan_absmax': float(b0['plan_absmax'][k]),
        'CL_plan_absmax': float(cl['plan_absmax'][k]),
        'B0_fmax': float(b0['fmax'][sl].max()), 'CL_fmax': float(cl['fmax'][sl].max()),
        'B0_cmd_mm': float(np.linalg.norm(b0['cmd'][sl], axis=1).sum() * 1000),
        'CL_cmd_mm': float(np.linalg.norm(cl['cmd'][sl], axis=1).sum() * 1000),
        'B0_disp_mm': float(np.linalg.norm(b0['disp'][sl], axis=1).sum() * 1000),
        'CL_disp_mm': float(np.linalg.norm(cl['disp'][sl], axis=1).sum() * 1000),
        'B0_track': float(np.linalg.norm(b0['disp'][sl].sum(0)) /
                          max(np.linalg.norm(b0['cmd'][sl].sum(0)), 1e-9)),
        'CL_track': float(np.linalg.norm(cl['disp'][sl].sum(0)) /
                          max(np.linalg.norm(cl['cmd'][sl].sum(0)), 1e-9)),
        'B0_grip_closed': int((b0['grip'][sl] < 0).sum()),
        'CL_grip_closed': int((cl['grip'][sl] < 0).sum()),
    })

gate = json.loads((OUT / 'identity_gate.json').read_text())
summary = {
    **MARK, 'scope': 'closed-loop simulator, single seed 17, B0 vs P2-clamp',
    'identity_gate': {k: v for k, v in gate.items() if k != 'per_chunk'},
    'first_force_divergence_frame': first_div,
    'B0': sb0, 'CL': scl,
    'comparison': {
        'cookies_delta': (scl['cookies_in_target'] or 0) - (sb0['cookies_in_target'] or 0),
        'force_absmax_ratio': scl['force_absmax'] / sb0['force_absmax'],
        'frames_gt_50N_ratio': (scl['frames_gt_50N'] / sb0['frames_gt_50N']
                               if sb0['frames_gt_50N'] else None),
        'jam_steps_ratio': (scl['jam_steps_cmd_gt20mm_track_lt0.3'] /
                            sb0['jam_steps_cmd_gt20mm_track_lt0.3']
                            if sb0['jam_steps_cmd_gt20mm_track_lt0.3'] else None),
        'plan_absmax_ratio': scl['plan_absmax_max'] / sb0['plan_absmax_max'],
    },
    'chunk_table': chunk_tbl,
    'trace_every_5_frames': rows,
}
(OUT / 'analysis_closed_loop.json').write_text(json.dumps(summary, indent=2, default=str))

lines = ['identity gate: %s (first divergent chunk %s)' % (
             summary['identity_gate']['gate_passed'],
             summary['identity_gate']['first_divergent_chunk']),
         '',
         '%-6s %-9s %-9s %-10s %-10s %-9s %-9s' % (
             'chunk', 'B0_plan', 'CL_plan', 'B0_fmax', 'CL_fmax', 'B0_track', 'CL_track')]
for c in chunk_tbl:
    lines.append('%-6d %-9.3f %-9.3f %-10.2f %-10.2f %-9.3f %-9.3f' % (
        c['chunk'], c['B0_plan_absmax'], c['CL_plan_absmax'],
        c['B0_fmax'], c['CL_fmax'], c['B0_track'], c['CL_track']))
lines += ['', 'B0 %s' % json.dumps(sb0), '', 'CL %s' % json.dumps(scl)]
(OUT / 'analysis_closed_loop.txt').write_text('\n'.join(lines) + '\n')
print('\n'.join(lines[:5]))
print('wrote', OUT / 'analysis_closed_loop.json')
