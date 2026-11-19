"""Scene-truth analysis of closed_loop_scene_20261118: did any cookie actually move with the gripper?

CPU only. Reads the per-arm npz written on the pod. No model, no simulator.
"""
import io
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'closed_loop_scene_20261118_v2'
CAL = json.load(io.open(str(ROOT.parent / 'step_040000' / 'force_calib_v1.json'), encoding='utf-8'))
CH = [int(c) for c in CAL['channels']]
MARK = {'NORMALIZER_IDENTITY': 'RECONSTRUCTED_DETERMINISTIC_NOT_ORIGINAL',
        'original_stats_sha256': 'UNKNOWN'}

TARGET_XY = np.array([0.095, 0.100])
SOURCE_XY = np.array([0.175, 0.315])
TABLE_Z = 0.750


def runs(mask, min_len=1):
    out, i = [], 0
    while i < len(mask):
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < len(mask) and mask[j]:
            j += 1
        if j - i >= min_len:
            out.append((i, j - 1))
        i = j
    return out


def load(path):
    z = np.load(str(path), allow_pickle=False)
    meta = json.loads(Path(str(path).replace('.npz', '.json')).read_text())
    n = len(z['step_opening'])
    cmd = np.zeros((n, 3))
    grip = np.zeros(n)
    for k in range((n + 9) // 10):
        key = 'chunk_%03d_adapter_out' % k
        if key not in z.files:
            break
        ad = np.asarray(z[key], float).reshape(-1, 14)
        m = min(10, n - 10 * k)
        cmd[10 * k:10 * k + m] = ad[:m, :3] * 0.025
        grip[10 * k:10 * k + m] = ad[:m, 6]
    return {'z': z, 'meta': meta, 'n': n, 'cmd': cmd, 'grip': grip,
            'eef': np.asarray(z['step_eef_pose'], float),
            'op': np.asarray(z['step_opening'], float),
            'force': np.asarray(z['step_force18'], float)[:, CH],
            'cp': np.asarray(z['step_cookie_pos'], np.float32),
            'in_t': np.asarray(z['step_in_target_mask'], bool),
            'in_s': np.asarray(z['step_in_source_mask'], bool),
            'src_n': np.asarray(z['step_cookies_in_source'], np.int64),
            'tgt_n': np.asarray(z['step_cookies'], np.int64)}


def analyse(arm):
    cp, eef = arm['cp'], arm['eef']
    T = cp.shape[0]
    base = cp[0]
    disp = np.linalg.norm(cp - base[None], axis=2)          # (T, C)
    max_disp = disp.max(0)                                   # per cookie
    moved = np.flatnonzero(max_disp > 0.003)
    # z profile
    zprof = cp[:, :, 2]
    lifted = np.flatnonzero((zprof - zprof[0][None]).max(0) > 0.005)

    carries = runs(arm['op'] < 0.30, 5)
    carry_rows = []
    for (s, e) in carries:
        seg_disp = np.linalg.norm(cp[e] - cp[s], axis=1)
        moving = np.flatnonzero(seg_disp > 0.003)
        ed = eef[e, :3] - eef[s, :3]
        close = np.linalg.norm(cp[s][moving][:, :2] - eef[s, :2][None], axis=1) if moving.size else np.array([])
        carry_rows.append({
            'frames': [int(s), int(e)],
            'op_min': float(arm['op'][s:e + 1].min()),
            'grip_cmd_mean': float(arm['grip'][s:e + 1].mean()),
            'eef_move_mm': (ed * 1000).round(1).tolist(),
            'cookies_moved_gt3mm': int(moving.size),
            'cookies_moved_ids': moving[:12].tolist(),
            'cookies_moved_seg_disp_mm': seg_disp[moving][:12].round(1).tolist()
            if moving.size else [],
            'start_dist_to_eef_mm': (close * 1000).round(1).tolist() if close.size else [],
            'cookie_z_start': cp[s][moving][:, 2].round(3).tolist() if moving.size else [],
            'cookie_z_end': cp[e][moving][:, 2].round(3).tolist() if moving.size else [],
        })

    # final resting place of the plates that ever moved
    final = cp[-1]
    fin = []
    for i in moved[:20]:
        fin.append({'id': int(i), 'start': base[i].round(3).tolist(), 'end': final[i].round(3).tolist(),
                    'z_max': float(zprof[:, i].max()),
                    'dist_to_target_xy_mm': float(np.linalg.norm(final[i, :2] - TARGET_XY) * 1000),
                    'in_target_final': bool(arm['in_t'][-1, i])})

    return {
        'frames': int(T),
        'cookie_count': int(cp.shape[1]),
        'cookies_moved_gt3mm': int(moved.size),
        'cookies_moved_ids': moved.tolist(),
        'cookies_max_disp_mm_top20': np.sort(max_disp)[::-1][:20].round(1).tolist(),
        'cookies_lifted_gt5mm': int(lifted.size),
        'in_target_counts': {'final': int(arm['in_t'][-1].sum()), 'max': int(arm['in_t'].sum(1).max())},
        'in_source_counts': {'start': int(arm['in_s'][0].sum()), 'final': int(arm['in_s'][-1].sum()),
                             'min': int(arm['in_s'].sum(1).min())},
        'sim_cookies_in_target_trace': {'max': int(arm['tgt_n'].max()), 'final': int(arm['tgt_n'][-1])},
        'sim_cookies_in_source_trace': {'start': int(arm['src_n'][0]), 'final': int(arm['src_n'][-1]),
                                        'min': int(arm['src_n'].min())},
        'carries': carry_rows,
        'moved_cookie_final_placements': fin,
        'record': arm['meta'].get('record'),
    }


files = sorted(OUT.glob('CL_s*.npz'))
if not files:
    raise SystemExit('no CL_s*.npz in %s' % OUT)

summary = {**MARK, 'scope': 'simulator closed loop, P2-clamped arm, scene-truth cookie logging', 'arms': {}}
lines = []
for f in files:
    arm = load(f)
    r = analyse(arm)
    summary['arms'][f.stem] = r
    lines.append('== %s  frames=%d cookies=%d moved>3mm=%d lifted>5mm=%d'
                 % (f.stem, r['frames'], r['cookie_count'], r['cookies_moved_gt3mm'], r['cookies_lifted_gt5mm']))
    lines.append('   target in-mask final/max %s   source in-mask start/final/min %s'
                 % (r['in_target_counts'], r['in_source_counts']))
    lines.append('   sim cookies_in_target max %d final %d ; cookies_in_source start %d final %d min %d'
                 % (r['sim_cookies_in_target_trace']['max'], r['sim_cookies_in_target_trace']['final'],
                    r['sim_cookies_in_source_trace']['start'], r['sim_cookies_in_source_trace']['final'],
                    r['sim_cookies_in_source_trace']['min']))
    lines.append('   top displacements mm %s' % r['cookies_max_disp_mm_top20'])
    for c in r['carries'][:6]:
        lines.append('   carry %s op_min %.3f grip %.2f eef_move %s moved %d %s'
                     % (c['frames'], c['op_min'], c['grip_cmd_mean'], c['eef_move_mm'],
                        c['cookies_moved_gt3mm'], c['cookies_moved_ids']))
        if c['cookies_moved_ids']:
            lines.append('        disp_mm %s  start_dist_to_eef_mm %s  z %s -> %s'
                         % (c['cookies_moved_seg_disp_mm'], c['start_dist_to_eef_mm'],
                            c['cookie_z_start'], c['cookie_z_end']))
    for p in r['moved_cookie_final_placements'][:6]:
        lines.append('   cookie %d start %s end %s zmax %.3f dist_to_target %.1f mm in_target %s'
                     % (p['id'], p['start'], p['end'], p['z_max'], p['dist_to_target_xy_mm'],
                        p['in_target_final']))
    lines.append('')

text = '\n'.join(lines)
io.open(str(OUT / 'analysis_scene.txt'), 'w', encoding='utf-8').write(text + '\n')
io.open(str(OUT / 'analysis_scene.json'), 'w', encoding='utf-8').write(
    json.dumps(summary, indent=2, default=str, ensure_ascii=False))
print(text)