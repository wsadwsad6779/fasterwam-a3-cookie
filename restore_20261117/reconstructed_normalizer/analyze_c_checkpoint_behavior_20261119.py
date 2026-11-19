"""Compact behaviour anatomy of an A/C evaluation directory (no model, no sim).

Reads episode_{A,C}_seed{N}.npz from one eval dir plus the expert reference
teacher_replay_20261119 (poses.npz + arrays.npz) and reports, per arm/seed:
gripper command sign timeline, measured opening, EEF start/final/range/path,
action magnitudes, and cookie/source counts. Output is small JSON so it can be
read back over the Jupyter contents API.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'c_class_eval_v2_step1000'
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path('/tmp/c_checkpoint_behavior.json')


def runs(sign):
    """List of (value, start, length) for a 1-D sign array."""
    out = []
    i = 0
    n = len(sign)
    while i < n:
        j = i
        while j < n and sign[j] == sign[i]:
            j += 1
        out.append([int(sign[i]), i, j - i])
        i = j
    return out


def arm_seed(path):
    d = np.load(path)
    act = np.asarray(d['action'], np.float64)
    eef = np.asarray(d['eef'], np.float64)
    opening = np.asarray(d['opening'], np.float64)
    cookies = np.asarray(d['cookies'], np.int64)
    source = np.asarray(d['source'], np.int64)
    pos = eef[:, :3]
    disp = np.linalg.norm(np.diff(pos, axis=0), axis=1) * 1000.0
    gi = act[:, 6] if act.size else np.zeros(0)
    sg = np.where(gi >= 0, 1, -1) if len(gi) else np.zeros(0, int)
    rs = runs(sg) if len(sg) else []
    close_runs = [r for r in rs if r[0] > 0]
    open_runs = [r for r in rs if r[0] < 0]
    return {
        'path': str(path),
        'frames': int(len(eef)), 'actions': int(len(act)),
        'gripper_cmd_plus_frac': float((sg > 0).mean()) if len(sg) else None,
        'gripper_cmd_flips': int((sg[1:] != sg[:-1]).sum()) if len(sg) > 1 else 0,
        'gripper_first_close_frame': int(next((r[1] for r in rs if r[0] > 0), -1)),
        'gripper_first_open_frame': int(next((r[1] for r in rs if r[0] < 0), -1)),
        'gripper_longest_close_run': max((r[2] for r in close_runs), default=0),
        'gripper_longest_open_run': max((r[2] for r in open_runs), default=0),
        'gripper_runs_head': rs[:12],
        'opening_start': float(opening[0]), 'opening_end': float(opening[-1]),
        'opening_min': float(opening.min()), 'opening_max': float(opening.max()),
        'opening_frames_lt_0p005': int((opening < 0.005).sum()),
        'opening_frames_gt_0p05': int((opening > 0.05).sum()),
        'eef_start': [round(float(v), 4) for v in pos[0]],
        'eef_final': [round(float(v), 4) for v in pos[-1]],
        'eef_min': [round(float(v), 4) for v in pos.min(0)],
        'eef_max': [round(float(v), 4) for v in pos.max(0)],
        'eef_path_mm': round(float(disp.sum()), 2),
        'eef_disp_mm_max': round(float(disp.max()), 2) if len(disp) else 0.0,
        'act_xyz_absmean': [round(float(np.abs(act[:, k]).mean()), 4) for k in range(3)]
        if len(act) else None,
        'act_xyz_absmax': [round(float(np.abs(act[:, k]).max()), 4) for k in range(3)]
        if len(act) else None,
        'act_rot_absmax': round(float(np.abs(act[:, 3:6]).max()), 4) if len(act) else None,
        'cookies_max': int(cookies.max()), 'cookies_final': int(cookies[-1]),
        'cookies_first_nonzero': int(np.argmax(cookies > 0)) if (cookies > 0).any() else -1,
        'source_start': int(source[0]), 'source_final': int(source[-1]),
        'source_min': int(source.min()),
    }


def expert_ref():
    P = ROOT / 'teacher_replay_20261119'
    if not (P / 'poses.npz').exists():
        return {'error': 'missing teacher_replay_20261119/poses.npz'}
    arr = np.load(P / 'arrays.npz', allow_pickle=True)
    starts = np.asarray(arr['ep_start'], np.int64)
    lens = np.asarray(arr['ep_len'], np.int64)
    a = arr['action']
    z = np.load(P / 'poses.npz')
    seeds = [int(k.split('_')[-1]) for k in z.files if k.startswith('poses_')]
    eps = []
    for k, seed in enumerate(seeds):
        poses = np.asarray(z['poses_%d' % seed], np.float64)
        s, n = int(starts[k]), int(lens[k])
        assert len(poses) == n + 1, (seed, len(poses), n)
        pos = poses[:n, :3]
        gi = a[s:s + n, 6]
        sg = np.where(gi >= 0, 1, -1)
        rs = runs(sg)
        eps.append({
            'ep': k, 'seed': seed, 'frames': n,
            'eef_start': [round(float(v), 4) for v in pos[0]],
            'eef_final': [round(float(v), 4) for v in pos[-1]],
            'eef_min': [round(float(v), 4) for v in pos.min(0)],
            'eef_max': [round(float(v), 4) for v in pos.max(0)],
            'path_mm': round(float(np.linalg.norm(np.diff(pos, axis=0), axis=1).sum() * 1000), 2),
            'gripper_cmd_plus_frac': round(float((sg > 0).mean()), 4),
            'gripper_flips': int((sg[1:] != sg[:-1]).sum()),
            'gripper_close_runs': [[int(r[1]), int(r[2])] for r in rs if r[0] > 0][:8],
            'gripper_open_runs': [[int(r[1]), int(r[2])] for r in rs if r[0] < 0][:8],
        })
    return {'episodes': eps}


def main():
    out = {'dir': str(DIR), 'expert_reference': expert_ref(), 'arms': {}}
    for arm in ('A', 'C'):
        for p in sorted(DIR.glob('episode_%s_seed*.npz' % arm)):
            seed = int(p.stem.split('seed')[-1])
            out['arms']['%s_seed%d' % (arm, seed)] = arm_seed(p)
    OUT.write_text(json.dumps(out, indent=1))
    print('WROTE', OUT)


if __name__ == '__main__':
    main()