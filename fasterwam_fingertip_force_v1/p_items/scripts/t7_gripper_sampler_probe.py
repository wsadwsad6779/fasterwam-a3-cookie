"""Verify V3 continuous gripper labels and the phase-anchored sampler.

V3 labels: column 6 is the raw expert commanded opening (1=open). The sampler is
balanced: 30% windows anchored at an open rise (>=0.8), 20% at a release rise
(>=0.35, i.e. 0.28->0.49), 50% uniform. The probe measures raw-space fractions,
edge coverage versus uniform sampling, and force targets/dropout.
"""
import json
import sys
import time

import numpy as np

sys.path.insert(0, '/workspace')

OUT = '/workspace/fwv1/p_items/GRIPPER_SAMPLER_PROBE_V3.json'
AH = 32


def covers(src, bb, attr):
    ep = int(bb['meta']['episode'])
    st = int(bb['meta']['start'])
    edges = getattr(src, attr)[ep]
    return bool(((edges >= st) & (edges < st + AH)).any())


def main():
    from t7_data import A3BatchSource
    t0 = time.time()
    src = A3BatchSource(split='train', batch_size=8, seed=123, balanced=True)
    usrc = A3BatchSource(split='train', batch_size=8, seed=123, balanced=False)
    n = 50
    open_fr, close_fr, release_fr, drop_fr = [], [], [], []
    open_cov, release_cov = [], []
    u_open_fr, u_release_fr, u_open_cov, u_release_cov = [], [], [], []
    per_mode = {}
    for _ in range(n):
        bb = src.next_batch()
        g = bb['grip_raw']
        of = float((g >= 0.8).float().mean())
        rf = float(((g > 0.35) & (g < 0.7)).float().mean())
        open_fr.append(of)
        close_fr.append(float((g <= 0.6).float().mean()))
        release_fr.append(rf)
        drop_fr.append(float(bb['force_dropped'].float().mean()))
        oc = covers(src, bb, 'phase_rise')
        rc = covers(src, bb, 'release_rise')
        open_cov.append(oc)
        release_cov.append(rc)
        md = per_mode.setdefault(bb['meta']['mode'], {'n': 0, 'open_frac': [], 'release_frac': [], 'open_cov': [], 'release_cov': []})
        md['n'] += 1
        md['open_frac'].append(of)
        md['release_frac'].append(rf)
        md['open_cov'].append(oc)
        md['release_cov'].append(rc)
        ub = usrc.next_batch()
        ug = ub['grip_raw']
        u_open_fr.append(float((ug >= 0.8).float().mean()))
        u_release_fr.append(float(((ug > 0.35) & (ug < 0.7)).float().mean()))
        u_open_cov.append(covers(usrc, ub, 'phase_rise'))
        u_release_cov.append(covers(usrc, ub, 'release_rise'))
    bb = src.next_batch()
    ep = int(bb['meta']['episode'])
    start = int(bb['meta']['start'])
    ls = int(src.lab_start[ep])
    raw = src.L[ls + start:ls + start + 5, 6]
    sign_check = {'episode': ep, 'start': start,
                  'raw_label_dim6_head': [round(float(x), 4) for x in raw],
                  'grip_raw_batch_head': [round(float(x), 4) for x in bb['grip_raw'][0, :5]],
                  'normalized_action_dim6_head': [round(float(x), 4) for x in bb['action'][0, :5, 6]],
                  'raw_match': bool(np.allclose(raw, bb['grip_raw'][0, :5].numpy(), atol=1e-6)),
                  'dataset_open_frac': float((src.L[:, 6] >= 0.8).mean()),
                  'dataset_close_frac': float((src.L[:, 6] <= 0.6).mean()),
                  'dataset_release_frac': float(((src.L[:, 6] > 0.35) & (src.L[:, 6] < 0.7)).mean()),
                  'force_target_mean': float(bb['force_target'].mean()),
                  'force_dropped_frac': float(bb['force_dropped'].float().mean())}
    out = {'worker': 'supervisor', 'ts': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
           'batches': n,
           'raw_open_frac_mean': float(np.mean(open_fr)),
           'raw_close_frac_mean': float(np.mean(close_fr)),
           'raw_release_frac_mean': float(np.mean(release_fr)),
           'force_drop_frac_mean': float(np.mean(drop_fr)),
           'open_edge_coverage': float(np.mean(open_cov)),
           'release_edge_coverage': float(np.mean(release_cov)),
           'uniform_baseline': {'open_frac_mean': float(np.mean(u_open_fr)),
                                'release_frac_mean': float(np.mean(u_release_fr)),
                                'open_edge_coverage': float(np.mean(u_open_cov)),
                                'release_edge_coverage': float(np.mean(u_release_cov))},
           'open_edge_gain_x': float(np.mean(open_cov) / max(1e-9, np.mean(u_open_cov))),
           'release_edge_gain_x': float(np.mean(release_cov) / max(1e-9, np.mean(u_release_cov))),
           'per_mode': {k: {'n': v['n'],
                            'open_frac_mean': float(np.mean(v['open_frac'])),
                            'release_frac_mean': float(np.mean(v['release_frac'])),
                            'open_edge_coverage': float(np.mean(v['open_cov'])),
                            'release_edge_coverage': float(np.mean(v['release_cov']))}
                        for k, v in per_mode.items()},
           'sign_check': sign_check,
           'elapsed_s': round(time.time() - t0, 1)}
    rm = out['per_mode'].get('release', {})
    om = out['per_mode'].get('open', {})
    out['verdict'] = ('pass' if (sign_check['raw_match'] and rm.get('n', 0) >= 5
                                 and rm.get('release_edge_coverage', 0) >= 0.9
                                 and om.get('open_frac_mean', 0) >= 0.20
                                 and np.mean(open_fr) >= 0.08)
                      else 'fail')
    json.dump(out, open(OUT, 'w'), indent=1)
    print(json.dumps(out, indent=1), flush=True)


if __name__ == '__main__':
    main()