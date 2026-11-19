"""Offline paired read of the C-class NPZ evidence (no model, no simulator)."""
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / 'c_class_eval_20261119'
NAMES = ['L_fx', 'L_fy', 'L_fz', 'pad_inner', 'pad_outer', 'gripper']


def first_cross(x, level):
    idx = np.where(np.abs(x) > level)[0]
    return int(idx[0]) if len(idx) else None


def stats(path):
    d = np.load(path)
    m = d['measured']
    eef = d['eef']
    act = d['action'] if d['action'].size else np.zeros((0, 7))
    op = d['opening']
    disp = np.linalg.norm(np.diff(eef[:, :3], axis=0), axis=1) * 1000.0
    cmd = np.abs(act[:len(disp), :3]).max(1) * 25.0 if len(disp) else np.zeros(0)
    track = np.divide(disp, cmd, out=np.ones_like(disp), where=cmd > 0)
    gi = act[:, 6] if act.size else np.zeros(0)
    return {
        'frames': int(len(m)),
        'force_absmax': float(np.abs(m).max()),
        'per_channel_absmax': np.abs(m).max(0).tolist(),
        'first_gt_50': [first_cross(m[:, i], 50) for i in range(6)],
        'first_gt_100': [first_cross(m[:, i], 100) for i in range(6)],
        'first_gt_150': [first_cross(m[:, i], 150) for i in range(6)],
        'eef_start_m': eef[0, :3].round(4).tolist(),
        'eef_end_m': eef[-1, :3].round(4).tolist(),
        'eef_span_m': (eef[:, :3].max(0) - eef[:, :3].min(0)).round(4).tolist(),
        'path_mm': float(disp.sum()),
        'jam_steps': int(((cmd > 20.0) & (track < 0.3)).sum()),
        'gripper_plus_frac': float((gi > 0).mean()) if len(gi) else None,
        'gripper_flips': int((np.sign(gi[1:]) != np.sign(gi[:-1])).sum()) if len(gi) > 1 else 0,
        'opening_min': float(op.min()), 'opening_max': float(op.max()),
        'cookies_final': int(d['cookies'][-1]), 'cookies_max': int(d['cookies'].max()),
        'source_final': int(d['source'][-1]),
        'cookies_any_frame': int((d['cookies'] > 0).sum()),
        'noise_seeds_unique': int(len(set(d['noise_seeds'].tolist()))),
    }


def main():
    out = {'channel_names': NAMES, 'episodes': {}}
    for arm in ('A', 'C'):
        for seed in (29, 30):
            p = OUT / ('episode_%s_seed%d.npz' % (arm, seed))
            if not p.exists():
                continue
            out['episodes']['%s_seed%d' % (arm, seed)] = stats(p)
    (OUT / 'offline_compare.json').write_text(json.dumps(out, indent=2))
    for k, v in out['episodes'].items():
        print(k, 'frames=%d absmax=%.1f path=%.0fmm jam=%d flips=%d plus=%.2f '
              'cookies_max=%d source=%d' % (v['frames'], v['force_absmax'], v['path_mm'],
                                            v['jam_steps'], v['gripper_flips'],
                                            v['gripper_plus_frac'] or -1,
                                            v['cookies_max'], v['source_final']))
    print(json.dumps(out['episodes']))


if __name__ == '__main__':
    main()
