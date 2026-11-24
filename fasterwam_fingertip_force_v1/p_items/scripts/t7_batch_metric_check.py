"""Batch-level check for C-01: raw gripper distribution, metric semantics, force
targets, dropout, and window bounds."""
import json
import sys
import time

import numpy as np

sys.path.insert(0, '/workspace')

OUT = '/workspace/fwv1/p_items/T7_BATCH_METRIC_CHECK.json'


def main():
    from t7_data import A3BatchSource, AH, HIST
    B = 8
    src = A3BatchSource(split='train', batch_size=B, seed=7, balanced=True)
    n = 60
    close, openf, release, drop, ftgt = [], [], [], [], []
    close_norm, open_norm = [], []
    raw_match = True
    last = None
    for _ in range(n):
        bb = src.next_batch()
        g = bb['grip_raw']
        close.append(float((g <= 0.6).float().mean()))
        openf.append(float((g >= 0.8).float().mean()))
        release.append(float(((g > 0.35) & (g < 0.7)).float().mean()))
        drop.append(float(bb['force_dropped'].float().mean()))
        ftgt.append(float(bb['force_target'].mean()))
        an = bb['action'][:, :, 6]
        close_norm.append(float((an <= 0.6).float().mean()))
        open_norm.append(float((an >= 0.8).float().mean()))
        last = bb
    ep = int(last['meta']['episode'])
    start = int(last['meta']['start'])
    ls = int(src.lab_start[ep])
    raw = src.L[ls + start:ls + start + 5, 6]
    raw_match = bool(np.allclose(raw, last['grip_raw'][0, :5].numpy(), atol=1e-6))
    bad_bounds = []
    for ep in src.episodes:
        frames = int(src.el[ep])
        max_start = frames - B - AH - 1
        if max_start < 1:
            bad_bounds.append({'episode': int(ep), 'frames': frames, 'max_start': int(max_start)})
    out = {'worker': 'supervisor', 'ts': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
           'batches': n, 'batch_size': B,
           'raw_space': {'close_frac_mean': float(np.mean(close)),
                         'open_frac_mean': float(np.mean(openf)),
                         'release_frac_mean': float(np.mean(release)),
                         'open_batches_gt10pct': int(sum(1 for x in openf if x > 0.10))},
           'normalized_space_old_metric': {'close_frac_mean': float(np.mean(close_norm)),
                                           'open_frac_mean': float(np.mean(open_norm))},
           'force': {'target_mean': float(np.mean(ftgt)),
                     'target_finite': bool(np.all(np.isfinite(ftgt))),
                     'dropout_frac_mean': float(np.mean(drop))},
           'grip_raw_matches_label': raw_match,
           'bad_episode_bounds': bad_bounds,
           'verdict': 'pass' if (raw_match and not bad_bounds and np.mean(openf) > 0.08
                                 and np.mean(release) > 0.25 and np.all(np.isfinite(ftgt))) else 'fail'}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(json.dumps(out, indent=1), flush=True)


if __name__ == '__main__':
    main()