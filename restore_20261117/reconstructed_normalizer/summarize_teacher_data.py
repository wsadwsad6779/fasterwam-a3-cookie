"""Summarise the collected teacher episodes (the run-time summary crashed on a formatting bug,
the npz data themselves are intact). CPU only."""
import glob
import json
import os

import numpy as np

ROOT = '/workspace/eval_restore_20261117/reconstructed_normalizer'
OUT = '/workspace/eval_restore_20261117/reconstructed_normalizer/TEACHER_DATA_SUMMARY.json'
res = {'NORMALIZER_IDENTITY': 'RECONSTRUCTED_DETERMINISTIC_NOT_ORIGINAL',
       'TRAINING_DATA_IDENTITY': 'CANDIDATE_TEACHER_DATA', 'episodes': {}}
files = sorted(glob.glob(ROOT + '/teacher*/teacher_s*.npz'))
for f in files:
    z = np.load(f)
    teach = z['teach']
    f18 = z['force18']
    pad = np.abs(f18[:, [12, 13]]).max(1)
    tc = z['cookies_in_target']
    sc = z['cookies_in_source']
    key = f.split('/teacher')[-2].split('/')[-1] + '/' + os.path.basename(f)[:-4]
    res['episodes'][key] = {
        'steps': int(len(teach)),
        'img_front_shape': list(z['img_front'].shape),
        'img_wrist_shape': list(z['img_wrist'].shape),
        'teach_counts': {str(k): int((teach == k).sum()) for k in (-2, -1, 0, 1)},
        'release_allowed_frames': np.flatnonzero(teach == 1)[:20].tolist(),
        'hold_high_frames': int((teach == -1).sum()),
        'final_cookies_in_target': int(tc[-1]),
        'max_cookies_in_target': int(tc.max()),
        'final_source': int(sc[-1]),
        'force_absmax': float(np.abs(f18).max()),
        'pad_absmax': float(pad.max()),
        'frames_pad_gt100': int((pad > 100).sum()),
        'frames_pad_gt500': int((pad > 500).sum()),
    }
with open(OUT, 'w') as fh:
    json.dump(res, fh, indent=2, default=str)
print(json.dumps(res, indent=1)[:2000])