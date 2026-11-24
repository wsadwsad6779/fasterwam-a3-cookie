"""Build corrected 7D labels V3: dim6 is the continuous commanded opening
(verified action[7], 1=open), replacing the pm1 threshold that collapsed
on-cookie carry (0.45) into +1 and hid the opening magnitude."""
import json
import os
import time

import numpy as np
import pyarrow.parquet as pq

P = '/workspace/fwv1/p_items'
PARQ = [P + '/hf_multi/data/chunk-000/file-000.parquet',
        P + '/hf_multi/data/chunk-000/file-001.parquet']
V2 = P + '/A3_845_7D_LABELS_V2.npz'
OUT_NPZ = P + '/A3_845_7D_LABELS_V3.npz'
OUT_JSON = P + '/GRIPPER_LABEL_V3_BUILD.json'
STATS_OUT = '/workspace/fwv1/data/a3_fw_845/dataset_stats_v3.json'
SPLIT = P + '/DATASET_845_SPLIT_MANIFEST_V2.json'


def rotvec(q0, q1):
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    w0, x0, y0, z0 = q0
    w1, x1, y1, z1 = q1
    cw, cx, cy, cz = w0, -x0, -y0, -z0
    w = w1 * cw - x1 * cx - y1 * cy - z1 * cz
    x = w1 * cx + x1 * cw + y1 * cz - z1 * cy
    y = w1 * cy - x1 * cz + y1 * cw + z1 * cx
    z = w1 * cz + x1 * cy - y1 * cx + z1 * cw
    if w < 0:
        w, x, y, z = -w, -x, -y, -z
    ang = 2 * np.arccos(np.clip(w, -1, 1))
    s = np.sqrt(max(1e-12, 1 - w * w))
    if ang < 1e-8:
        return np.zeros(3)
    return np.array([x, y, z]) / s * ang


def main():
    ts = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    z2 = np.load(V2)
    el = z2['ep_len'].astype(np.int64)
    L2 = z2['label'].astype(np.float32)
    ls2 = z2['ep_start_label'].astype(np.int64)
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
    labels = []
    lab_start = []
    frame_start = []
    max_dim6_diff = 0.0
    cursor = 0
    for k, ep in enumerate(ids):
        s, e = int(st[k]), int(en[k])
        n = e - s
        g = A[s:e, 7]
        lab = np.zeros((n - 1, 7), np.float32)
        lab[:, :6] = L2[int(ls2[k]):int(ls2[k]) + n - 1, :6]
        lab[:, 6] = g[:n - 1]  # continuous commanded opening, 1=open
        labels.append(lab)
        lab_start.append(cursor)
        frame_start.append(s)
        cursor += len(lab)
        max_dim6_diff = max(max_dim6_diff, float(np.abs(lab[:, 6] - g[:n - 1]).max()))
    L = np.concatenate(labels, 0).astype(np.float32)
    np.savez_compressed(OUT_NPZ, label=L,
                        ep_start_label=np.asarray(lab_start, np.int64),
                        ep_start_frame=np.asarray(frame_start, np.int64),
                        ep_len=el)
    split = json.load(open(SPLIT))
    train_ids = set(int(x) for x in split['train_episodes'])
    rows = []
    for k in range(len(lab_start)):
        if k in train_ids:
            s = int(lab_start[k])
            rows.append(L[s:s + int(el[k])])
    Atr = np.concatenate(rows, 0)
    gtr = Atr[:, 6]
    stats = {'action': {'default': {
        'global_min': [float(x) for x in Atr.min(0)],
        'global_max': [float(x) for x in Atr.max(0)]}},
        'state': json.load(open('/workspace/fwv1/data/a3_fw_845/dataset_stats.json'))['state'],
        'meta': {'split': SPLIT, 'train_episodes': len(train_ids), 'rows': int(len(Atr)),
                 'gripper': 'continuous commanded opening (1=open), min/max of train split',
                 'labels': OUT_NPZ}}
    json.dump(stats, open(STATS_OUT, 'w'), indent=2)
    evidence = {'worker': 'task7', 'ts': ts,
                'labels_v3': OUT_NPZ,
                'dim6_semantics': 'continuous commanded opening (verified action[7], 1=open)',
                'max_dim6_vs_raw_action7_diff': max_dim6_diff,
                'train_stats': {'gripper_min': float(gtr.min()), 'gripper_max': float(gtr.max()),
                                'gripper_mean': float(gtr.mean()),
                                'open_ge_0.8_frac': float((gtr >= 0.8).mean()),
                                'partial_0.35_0.8_frac': float(((gtr > 0.35) & (gtr < 0.8)).mean()),
                                'closed_le_0.35_frac': float((gtr <= 0.35).mean())},
                'v2_pm1_close_frac': 0.935,
                'note': 'V2 collapsed carry opening ~0.45 into +1 (close). V3 keeps the exact opening value, so the model can learn open/partial/closed magnitudes instead of a constant +1.',
                'stats_file': STATS_OUT}
    json.dump(evidence, open(OUT_JSON, 'w'), indent=1)
    print(json.dumps({k: evidence[k] for k in ['max_dim6_vs_raw_action7_diff', 'train_stats']}))


if __name__ == '__main__':
    main()