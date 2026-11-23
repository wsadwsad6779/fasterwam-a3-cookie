import json
import os
import time

import numpy as np
import pyarrow.parquet as pq

SPLIT = '/workspace/fwv1/p_items/DATASET_800_SPLIT_MANIFEST.json'
LABELS = '/workspace/fwv1/p_items/A3_845_7D_LABELS_V2.npz'
PARQ = ['/workspace/fwv1/p_items/hf_multi/data/chunk-000/file-000.parquet',
        '/workspace/fwv1/p_items/hf_multi/data/chunk-000/file-001.parquet']
OUT_DIR = '/workspace/fwv1/data/a3_fw_845'
OUT = os.path.join(OUT_DIR, 'dataset_stats.json')
EV = '/workspace/fwv1/p_items/DATASET_845_NORM_STATS.json'


def main():
    split = json.load(open(SPLIT))
    train_ids = set(int(x) for x in split['train_episodes'])
    z = np.load(LABELS)
    L = z['label'].astype(np.float64)
    lab_start = z['ep_start_label'].astype(np.int64)
    el = z['ep_len'].astype(np.int64)
    action_rows = []
    for k in range(len(lab_start)):
        s = int(lab_start[k])
        action_rows.append(L[s:s + int(el[k])])
    A = np.concatenate(action_rows, 0)
    t = pq.read_table(PARQ, columns=['episode_index', 'observation.state'])
    E = np.asarray(t['episode_index'])
    S = np.asarray(t['observation.state'].to_pylist(), np.float64)
    mask = np.isin(E, list(train_ids))
    Str = S[mask]
    stats = {
        'action': {'default': {'global_min': action_train_min(A, split, lab_start, el, train_ids),
                               'global_max': action_train_max(A, split, lab_start, el, train_ids)}},
        'state': {'default': {'global_min': [float(x) for x in Str.min(0)],
                              'global_max': [float(x) for x in Str.max(0)]}},
        'meta': {'n_train_episodes': len(train_ids), 'action_rows_total': int(A.shape[0]),
                 'state_rows_train': int(Str.shape[0]),
                 'action_dims': 7, 'state_dims': 8,
                 'source': 'train split only', 'labels': LABELS, 'parquet': PARQ},
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    json.dump(stats, open(OUT, 'w'), indent=2)
    ev = dict(stats)
    ev['ts'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    json.dump(ev, open(EV, 'w'), indent=2)
    print(json.dumps({'action_min': stats['action']['default']['global_min'],
                      'action_max': stats['action']['default']['global_max'],
                      'state_min': stats['state']['default']['global_min'],
                      'state_max': stats['state']['default']['global_max']}))


def _train_action(A, split, lab_start, el, train_ids):
    parts = []
    for k in range(len(lab_start)):
        ep = k
        if ep not in train_ids:
            continue
        s = int(lab_start[k])
        parts.append(A[s:s + int(el[k])])
    return np.concatenate(parts, 0)


def action_train_min(A, split, lab_start, el, train_ids):
    return [float(x) for x in _train_action(A, split, lab_start, el, train_ids).min(0)]


def action_train_max(A, split, lab_start, el, train_ids):
    return [float(x) for x in _train_action(A, split, lab_start, el, train_ids).max(0)]


if __name__ == '__main__':
    main()