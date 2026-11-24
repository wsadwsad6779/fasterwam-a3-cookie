"""Near-duplicate audit for the 845-episode cookie dataset.

Layers:
  1) action trajectory (7 arm dims + grip, resampled to 32 points)
  2) eef pose trajectory (14 dims, resampled to 16 points)
  3) layout (recorded cookie x/y positions for the 749 identity-known episodes)

For each layer: nearest-neighbor distance distribution, pair counts under
thresholds, and cross-split pairs under the same thresholds. This closes the
"same-trajectory re-render / layout-contact near-dup" layers that the exact
sha1 dedup in t7_d2_audit.py cannot see.
"""
import json
import os
import time

import numpy as np
import pyarrow.parquet as pq

P = '/workspace/fwv1/p_items'
META = P + '/hf_multi/a3_episode_metadata.jsonl'
PARQ = [P + '/hf_multi/data/chunk-000/file-000.parquet',
        P + '/hf_multi/data/chunk-000/file-001.parquet']
SPLIT = P + '/DATASET_845_SPLIT_MANIFEST_V2.json'
OUT = P + '/DATASET_845_NEAR_DUP_AUDIT.json'
N_A = 32
N_X = 16


def resample(a, n):
    a = np.asarray(a, np.float64)
    if len(a) == 0:
        return np.zeros((n, a.shape[1] if a.ndim > 1 else 1), np.float64)
    if len(a) == 1:
        return np.repeat(a, n, axis=0)
    src = np.linspace(0.0, 1.0, len(a))
    dst = np.linspace(0.0, 1.0, n)
    return np.stack([np.interp(dst, src, a[:, j]) for j in range(a.shape[1])], 1)


def nn_stats(F, names, split, thresholds):
    sq = (F * F).sum(1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (F @ F.T)
    np.maximum(d2, 0.0, out=d2)
    d = np.sqrt(d2)
    np.fill_diagonal(d, np.inf)
    nn = d.min(1)
    out = {'n': int(len(F)), 'nn_min': float(nn.min()), 'nn_p50': float(np.median(nn)),
           'nn_p90': float(np.quantile(nn, 0.9)), 'nn_max': float(nn.max())}
    for th in thresholds:
        m = d < th
        pairs = int(m.sum() // 2)
        cross = 0
        idx = np.argwhere(np.triu(m, 1))
        for i, j in idx:
            if split[i] != split[j]:
                cross += 1
        out['pairs_lt_%g' % th] = pairs
        out['cross_split_pairs_lt_%g' % th] = int(cross)
    out['names'] = names
    return out


def main():
    t0 = time.time()
    meta = {}
    for line in open(META):
        if not line.strip():
            continue
        d = json.loads(line)
        meta.setdefault(int(d['episode_index']), []).append(d)
    t = pq.read_table(PARQ, columns=['episode_index', 'action', 'observation.eef_pose'])
    E = np.asarray(t['episode_index'])
    A = np.asarray(t['action'].to_pylist(), np.float64)
    X = np.asarray(t['observation.eef_pose'].to_pylist(), np.float64)
    order = np.argsort(E, kind='stable')
    E, A, X = E[order], A[order], X[order]
    b = np.flatnonzero(np.diff(E)) + 1
    st = np.r_[0, b]
    en = np.r_[b, len(E)]
    ids = [int(E[st[k]]) for k in range(len(st))]
    sp = json.load(open(SPLIT))
    split_map = {}
    for name in ('train', 'val', 'test'):
        for ep in sp[name + '_episodes']:
            split_map[int(ep)] = name
    split = [split_map.get(i, '?') for i in ids]

    FA = []
    FX = []
    for k, ep in enumerate(ids):
        s, e = int(st[k]), int(en[k])
        a = A[s:e]
        FA.append(resample(a, N_A).ravel())
        FX.append(resample(X[s:e], N_X).ravel())
    FA = np.asarray(FA)
    FX = np.asarray(FX)

    action_stats = nn_stats(FA, ['action'] * len(FA), split, [0.02, 0.05, 0.1, 0.2])
    eef_stats = nn_stats(FX, ['eef'] * len(FX), split, [0.002, 0.005, 0.01, 0.02])

    layouts = []
    layout_eps = []
    for k, ep in enumerate(ids):
        ents = meta.get(ep) or []
        n = int(en[k] - st[k])
        cand = [x for x in ents if int(x.get('frames', -1)) == n and '2x5' in str(x.get('source_dataset'))]
        if not cand:
            cand = [x for x in ents if int(x.get('frames', -1)) == n]
        if not cand:
            continue
        rnd = (cand[0].get('randomization') or {})
        cp = rnd.get('cookie_positions')
        if not cp:
            continue
        arr = np.asarray(cp, np.float64)
        layouts.append(arr[:, :2].ravel())
        layout_eps.append(ep)
    layout_stats = None
    if layouts:
        L = np.asarray(layouts)
        split_l = [split_map.get(i, '?') for i in layout_eps]
        layout_stats = nn_stats(L, [str(e) for e in layout_eps], split_l, [0.001, 0.005, 0.01, 0.02])
        layout_stats['cookies_per_layout'] = int(np.asarray(json.loads(open(META).readline())['randomization']['cookie_positions']).shape[0]) if layouts else None

    result = {'worker': 'supervisor', 'ts': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
              'purpose': 'multi-layer near-duplicate / leakage audit beyond exact sha1',
              'action_layer': action_stats, 'eef_layer': eef_stats, 'layout_layer': layout_stats,
              'thresholds_note': 'action L2 in radians over 32x8 resampled trajectory; eef L2 in metres over 16x14; layout L2 in metres over flatten cookie x/y',
              'elapsed_s': round(time.time() - t0, 1)}
    json.dump(result, open(OUT, 'w'), indent=1)
    print(json.dumps(result, indent=1)[:6000], flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()