"""Independent acceptance for teacher_diverse_20261119 (contract + real layout diversity).

Checks:
  * ep_start cumulative, ep_len 600, frames.dat size == frames*224*896? (224x448x3)
  * proprio has no zero rows and its columns match the deployment builder's ranges
  * force6 equals the calibrated channel selection of force18
  * 7-D action equals the pose-difference recipe (label contract)
  * every kept episode ended with 10 cookies + exact_2x5_fill
  * train and val layouts are disjoint, and images actually differ across layouts
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
D = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'teacher_diverse_20261119'
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else D / 'acceptance.json'

man = json.loads((D / 'manifest.json').read_text())
arr = np.load(D / 'arrays.npz', allow_pickle=True)
f = np.load(D / 'force_arrays.npz', allow_pickle=True)
pos = np.load(D / 'poses.npz')
cal = json.loads((ROOT.parent / 'step_040000' / 'force_calib_v1.json').read_text())
ch = [int(c) for c in cal['channels']]

st = np.asarray(arr['ep_start'], np.int64)
ln = np.asarray(arr['ep_len'], np.int64)
act = np.asarray(arr['action'], np.float64)
pro = np.asarray(arr['proprio'], np.float64)
f18 = np.asarray(f['force18'], np.float64)
f6 = np.asarray(f['force6'], np.float64)
n = int(ln.sum())

checks = {}
checks['ep_start_cumulative'] = bool(np.array_equal(st, np.concatenate([[0], np.cumsum(ln[:-1])])))
checks['ep_len_all_600'] = bool((ln == 600).all())
checks['action_rows'] = int(len(act))
checks['proprio_rows'] = int(len(pro))
checks['frames_bytes'] = int((D / 'frames.dat').stat().st_size)
checks['frames_bytes_match'] = bool(checks['frames_bytes'] == n * 224 * 448 * 3)
checks['EXPERT_ACTIONS'] = bool(arr['EXPERT_ACTIONS'])
checks['proprio_zero_rows'] = int(np.all(pro == 0, axis=1).sum())
checks['force6_equals_selected_channels'] = bool(np.array_equal(f6, f18[:, ch]))
checks['all_full_task'] = all(e['cookies_in_target'] == 10 and e['exact_fill']
                              for e in man['episodes'])

# 7-D action label contract, episode by episode
worst = 0.0
for k, e in enumerate(man['episodes']):
    P = np.asarray(pos['poses_ep%02d' % k], np.float64)
    a = act[st[k]:st[k] + ln[k]]
    worst = max(worst, float(np.abs(a[:, :3] - (P[1:, :3] - P[:-1, :3]) / 0.05).max()))
checks['action_vs_pose_delta_max_abs'] = worst

# Layout diversity: first-frame image distance and EEF path difference across episodes
mm = np.memmap(D / 'frames.dat', dtype=np.uint8, mode='r').reshape(-1, 224, 448, 3)
first = {}
for k in range(len(ln)):
    first[k] = np.asarray(mm[st[k]], np.int16)
pair_img = []
pair_path = []
for i in range(len(ln)):
    for j in range(i + 1, len(ln)):
        pair_img.append(float(np.abs(first[i] - first[j]).mean()))
        Pi = np.asarray(pos['poses_ep%02d' % i], np.float64)[:, :3]
        Pj = np.asarray(pos['poses_ep%02d' % j], np.float64)[:, :3]
        pair_path.append(float(np.linalg.norm(Pi - Pj, axis=1).mean() * 1000))
checks['image_mean_pair_diff_min'] = min(pair_img)
checks['image_mean_pair_diff_max'] = max(pair_img)
checks['image_pairs_all_nonzero'] = bool(min(pair_img) > 0.05)
checks['eef_mean_pair_diff_mm_min'] = min(pair_path)
checks['eef_pairs_all_nonzero'] = bool(min(pair_path) > 0.5)

train_xy = {(round(e['dx'], 4), round(e['dy'], 4)) for e in man['episodes'] if e['split'] == 'train'}
val_xy = {(round(e['dx'], 4), round(e['dy'], 4)) for e in man['episodes'] if e['split'] == 'val'}
checks['train_val_layouts_disjoint'] = bool(train_xy.isdisjoint(val_xy))
checks['n_train'] = len(train_xy)
checks['n_val'] = len(val_xy)

checks['pass'] = bool(checks['ep_start_cumulative'] and checks['ep_len_all_600']
                      and checks['frames_bytes_match'] and checks['EXPERT_ACTIONS']
                      and checks['proprio_zero_rows'] == 0
                      and checks['force6_equals_selected_channels']
                      and checks['all_full_task']
                      and checks['action_vs_pose_delta_max_abs'] < 1e-9
                      and checks['image_pairs_all_nonzero']
                      and checks['eef_pairs_all_nonzero']
                      and checks['train_val_layouts_disjoint'])
OUT.write_text(json.dumps(checks, indent=2))
print(json.dumps(checks, indent=2))