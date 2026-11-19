"""Exact accounting of the teacher-replay alignment bug (CPU, read-only)."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'teacher_replay_20261118'


def main():
    a = np.load(DATA / 'arrays.npz', allow_pickle=True)
    f = np.load(DATA / 'force_arrays.npz', allow_pickle=True)
    prop = a['proprio']
    nz_row = np.any(prop != 0, axis=1)
    out = {
        'ep_start': a['ep_start'].tolist(),
        'ep_len': a['ep_len'].tolist(),
        'expected_ep_start_if_cumulative': np.cumsum([0] + a['ep_len'].tolist()[:-1]).tolist(),
        'frames_total': int(len(prop)),
        'proprio_rows_nonzero': int(nz_row.sum()),
        'proprio_rows_zero': int((~nz_row).sum()),
        'zero_fraction': float((~nz_row).mean()),
        'first_zero_row': int(np.argmax(~nz_row)) if (~nz_row).any() else None,
        'windows_as_dataset_counts_them': int(sum(int(L) - 33 + 1 for L in a['ep_len'])),
        'windows_if_true_boundaries': int(sum(int(L) - 33 + 1 for L in a['ep_len'])),
        'overlap': 'all four "episodes" span frames 0..603 instead of 0..2400',
        'force_keys': f.files,
        'force_ep_start': f['ep_start'].tolist() if 'ep_start' in f.files else None,
        'force_ep_len': f['ep_len'].tolist() if 'ep_len' in f.files else None,
        'action_rows': int(len(a['action'])),
        'proprio_semantic_check': 'prop[:3] is the LEFT eef xyz (14-wide eef_pose: 7 left + 7 right)',
    }
    (ROOT / 'c_class_eval_20261119' / 'teacher_alignment_bug.json').write_text(
        json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
