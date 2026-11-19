"""Audit the teacher-replay training data against the deployed input contract.

Checks: action provenance (7th dim recipe), episode/window accounting, train/test isolation,
and the proprio-14 layout against what deployment actually feeds.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'teacher_replay_20261118'
RECON = ROOT / 'dataset_stats_14d_RECONSTRUCTED.json'


def col(a):
    return {'mean': float(np.mean(a)), 'std': float(np.std(a)),
            'min': float(np.min(a)), 'max': float(np.max(a)),
            'zeros_frac': float(np.mean(a == 0))}


def main():
    arr = np.load(DATA / 'arrays.npz', allow_pickle=True)
    fo = np.load(DATA / 'force_arrays.npz', allow_pickle=True)
    act = arr['action'].astype(np.float64)
    prop = arr['proprio'].astype(np.float64)
    starts = arr['ep_start']
    lens = arr['ep_len']
    stats = json.loads(RECON.read_text())

    n = len(act)
    windows = int(sum(int(L) - 33 + 1 for L in lens))
    grip = act[:, 6]
    flips = int((np.sign(grip[1:]) != np.sign(grip[:-1])).sum())
    # 7th dim vs hysteretic rule on the stored proprio surrogate (0.04*opening at col 6)
    dp = np.diff(prop[:, 6])
    close_frames = dp < -0.0008
    open_frames = dp > 0.0008
    label_vs_close = float(np.mean(grip[:-1][close_frames] > 0)) if close_frames.any() else None
    label_vs_open = float(np.mean(grip[:-1][open_frames] < 0)) if open_frames.any() else None

    st = stats['state']['default']
    am = stats['action']['default']
    pmean = np.asarray(st['global_mean'], dtype=np.float64).reshape(-1)
    pstd = np.asarray(st['global_std'], dtype=np.float64).reshape(-1)
    per_col = [col(prop[:, i]) for i in range(14)]
    zmismatch = []
    for i in range(14):
        sd = float(np.std(prop[:, i]))
        if pstd.shape[0] > i and pstd[i] > 1e-6:
            zmismatch.append(abs(float(np.mean(prop[:, i])) - float(pmean[i])) / float(pstd[i]))

    out = {
        'data_dir': str(DATA),
        'EXPERT_ACTIONS': bool(arr['EXPERT_ACTIONS']) if 'EXPERT_ACTIONS' in arr.files else None,
        'frames': int(n), 'action_shape': list(act.shape), 'proprio_shape': list(prop.shape),
        'episodes': len(starts), 'ep_lens': [int(x) for x in lens],
        'windows_num_frames33': windows,
        'force_frames': int(fo['force18'].shape[0]),
        'action_dims': [col(act[:, i]) for i in range(6)],
        'action_trans_norm_mean': float(np.linalg.norm(act[:, :3], axis=1).mean()),
        'action_clip_frac': float(np.mean(np.abs(act[:, :6]) >= 0.999)),
        'grip_plus_frac': float(np.mean(grip > 0)),
        'grip_flips': flips,
        'grip_label_follows_close_rule': label_vs_close,
        'grip_label_follows_open_rule': label_vs_open,
        'proprio_cols': per_col,
        'proprio_cols_8_to_13_all_zero': bool(np.all(prop[:, 8:14] == 0)),
        'proprio_cols_6_7_are_mirror_pairs': bool(np.allclose(prop[:, 6], -prop[:, 7])),
        'recon_proprio_mean': pmean.tolist(), 'recon_proprio_std': pstd.tolist(),
        'recon_action_mean': np.asarray(am['global_mean']).reshape(-1).tolist(),
        'recon_action_std': np.asarray(am['global_std']).reshape(-1).tolist(),
        'recon_proprio_min': np.asarray(st['global_min']).reshape(-1).tolist(),
        'recon_proprio_max': np.asarray(st['global_max']).reshape(-1).tolist(),
        'max_abs_z_of_data_column_mean_vs_recon_mean': (max(zmismatch) if zmismatch else None),
        'deploy_layout': 'base8 (eef xyz 3 + axis-angle 3 + two finger joints) + '
                         'six force-tare residuals (channels 0,1,2,12,13,16)',
        'data_layout': 'xyz 3 + rotvec 3 + (+0.04*opening, -0.04*opening) + six zero columns',
        'layout_mismatch': 'TRAINING_PROPRIO_COLUMNS_8_13_ZERO_AND_6_7_NOT_JOINTS',
        'seed_isolation': {'teacher_episode_seeds_unknown_here': True,
                           'test_seeds_used': [29, 30],
                           'window_overlap': 'episodes are contiguous; windows overlap within '
                                             'episode by design (num_frames=33)'},
        'val_split_note': 'train_lora_300.py used val_set_proportion=0 and idx_val=last 8 '
                          'windows of the same episodes -> val loss is NOT independent',
    }
    (ROOT / 'c_class_eval_20261119' / 'training_data_audit.json').write_text(
        json.dumps(out, indent=2, default=str))
    print(json.dumps({k: out[k] for k in (
        'frames', 'episodes', 'windows_num_frames33', 'EXPERT_ACTIONS', 'grip_plus_frac',
        'grip_flips', 'grip_label_follows_close_rule', 'grip_label_follows_open_rule',
        'proprio_cols_8_to_13_all_zero', 'max_abs_z_of_data_column_mean_vs_recon_mean',
        'layout_mismatch')}, indent=2))


if __name__ == '__main__':
    main()
