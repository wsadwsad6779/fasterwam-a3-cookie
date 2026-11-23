import json
import os
import shutil
import time

import numpy as np
import pyarrow.parquet as pq

SPLIT = '/workspace/fwv1/p_items/DATASET_845_SPLIT_MANIFEST_V2.json'
LABELS = '/workspace/fwv1/p_items/A3_845_7D_LABELS_V2.npz'
PARQ = ['/workspace/fwv1/p_items/hf_multi/data/chunk-000/file-000.parquet',
        '/workspace/fwv1/p_items/hf_multi/data/chunk-000/file-001.parquet']
OUT_DIR = '/workspace/fwv1/data/a3_fw_845'
OUT = os.path.join(OUT_DIR, 'dataset_stats.json')
EV = '/workspace/fwv1/p_items/DATASET_845_NORM_STATS_V2.json'


def main():
    split = json.load(open(SPLIT))
    train_ids = set(int(x) for x in split['train_episodes'])
    z = np.load(LABELS)
    L = z['label'].astype(np.float64)
    lab_start = z['ep_start_label'].astype(np.int64)
    el = z['ep_len'].astype(np.int64)
    parts = []
    for k in range(len(lab_start)):
        if k in train_ids:
            s = int(lab_start[k])
            parts.append(L[s:s + int(el[k])])
    A = np.concatenate(parts, 0)
    t = pq.read_table(PARQ, columns=['episode_index', 'observation.state'])
    E = np.asarray(t['episode_index'])
    S = np.asarray(t['observation.state'].to_pylist(), np.float64)
    mask = np.isin(E, list(train_ids))
    Str = S[mask]
    stats = {'action': {'default': {'global_min': [float(x) for x in A.min(0)],
                                    'global_max': [float(x) for x in A.max(0)]}},
             'state': {'default': {'global_min': [float(x) for x in Str.min(0)],
                                   'global_max': [float(x) for x in Str.max(0)]}},
             'meta': {'split': SPLIT, 'n_train_episodes': len(train_ids),
                      'action_rows': int(A.shape[0]), 'state_rows': int(Str.shape[0]),
                      'source': 'V2 train split only'}}
    os.makedirs(OUT_DIR, exist_ok=True)
    if os.path.exists(OUT):
        shutil.copyfile(OUT, OUT + '.v1_675.bak')
    json.dump(stats, open(OUT, 'w'), indent=2)
    ev = dict(stats)
    ev['ts'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    json.dump(ev, open(EV, 'w'), indent=2)
    print(json.dumps({'n_train': len(train_ids), 'action_min': stats['action']['default']['global_min'],
                      'action_max': stats['action']['default']['global_max'],
                      'state_min': stats['state']['default']['global_min'],
                      'state_max': stats['state']['default']['global_max']}))


if __name__ == '__main__':
    main()