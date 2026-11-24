"""Dataset-side phase statistics for the action-collapse diagnosis."""
import json
import time

import numpy as np

P = '/workspace/fwv1/p_items'
SPLIT = P + '/DATASET_845_SPLIT_MANIFEST_V2.json'
LABELS = P + '/A3_845_7D_LABELS_V2.npz'
OUT = P + '/ACTION_COLLAPSE_DIAG_DATASET.json'
ids = set(int(x) for x in json.load(open(SPLIT))['train_episodes'])
z = np.load(LABELS)
L = z['label'].astype(np.float64)
start = z['ep_start_label'].astype(np.int64)
el = z['ep_len'].astype(np.int64)
open_frames = []
close_frames = []
for k, (s, n) in enumerate(zip(start, el)):
    if k not in ids:
        continue
    seg = L[int(s):int(s) + int(n)]
    pm = seg[:, 6]
    dp = np.abs(seg[:, :3]).max(axis=1)
    open_frames.append(dp[pm < 0])
    close_frames.append(dp[pm > 0])
op = np.concatenate([x for x in open_frames if len(x)]) if open_frames else np.zeros(0)
cl = np.concatenate([x for x in close_frames if len(x)]) if close_frames else np.zeros(0)
out = {'worker': 'task7', 'ts': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
       'train_episodes': len(ids),
       'open_frames': int(len(op)), 'close_frames': int(len(cl)),
       'open_frac': float(len(op) / max(1, len(op) + len(cl))),
       'dp_absmax_open': float(op.max()) if len(op) else None,
       'dp_absmax_close': float(cl.max()) if len(cl) else None,
       'dp_mean_open': float(op.mean()) if len(op) else None,
       'dp_mean_close': float(cl.mean()) if len(cl) else None,
       'dp_nonzero_frac_open': float((op > 1e-4).mean()) if len(op) else None,
       'dp_nonzero_frac_close': float((cl > 1e-4).mean()) if len(cl) else None,
       'gripper_from_labels': {'close_frac': float(len(cl) / max(1, len(op) + len(cl))),
                               'open_frac': float(len(op) / max(1, len(op) + len(cl))) }}
json.dump(out, open(OUT, 'w'), indent=1)
print(json.dumps(out))