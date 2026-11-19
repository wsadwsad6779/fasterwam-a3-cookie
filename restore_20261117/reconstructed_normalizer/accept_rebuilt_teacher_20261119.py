"""Acceptance checks for the rebuilt teacher data (CPU, read-only)."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'teacher_replay_20261119'
RECON = ROOT / 'dataset_stats_14d_RECONSTRUCTED.json'
OUT = ROOT / 'c_class_eval_20261119' / 'data_acceptance_v2.json'


def main():
    a = np.load(DATA / 'arrays.npz', allow_pickle=True)
    f = np.load(DATA / 'force_arrays.npz', allow_pickle=True)
    p = np.load(DATA / 'poses.npz', allow_pickle=True)
    st = json.load(open(RECON))['state']['default']
    pmean = np.asarray(st['global_mean'], np.float64)
    pstd = np.asarray(st['global_std'], np.float64)
    prop = a['proprio'].astype(np.float64)
    act = a['action'].astype(np.float64)
    starts = [int(x) for x in a['ep_start']]
    lens = [int(x) for x in a['ep_len']]
    checks = {}
    checks['ep_start'] = starts
    checks['ep_start_cumulative'] = starts == list(np.cumsum([0] + lens[:-1]))
    checks['ep_len'] = lens
    checks['proprio_zero_rows'] = int(np.sum(np.all(prop == 0, axis=1)))
    checks['windows_total'] = int(sum(L - 33 + 1 for L in lens))
    checks['frames_dat_bytes'] = (DATA / 'frames.dat').stat().st_size
    checks['frames_dat_frame_count'] = checks['frames_dat_bytes'] // (224 * 448 * 3)
    checks['force_frames'] = int(f['force18'].shape[0])
    cal = json.load(open(ROOT.parent / 'step_040000' / 'force_calib_v1.json'))
    fc = [int(c) for c in cal['channels']]
    checks['force6_equals_selected_channels'] = bool(
        np.array_equal(f['force6'], f['force18'][:, fc]))
    z = (prop.mean(0) - pmean) / np.where(pstd > 1e-9, pstd, np.nan)
    checks['proprio_mean'] = prop.mean(0).round(5).tolist()
    checks['proprio_std'] = prop.std(0).round(5).tolist()
    checks['mean_abs_z_vs_recon'] = np.abs(z).round(3).tolist()
    checks['max_abs_z_of_column_mean'] = float(np.nanmax(np.abs(z)))
    # action vs pose deltas on sampled frames of each episode
    deltas = []
    for k, seed in enumerate([41, 42, 43, 44]):
        poses = p['poses_%d' % seed]
        opens = p['opening_%d' % seed]
        s0 = starts[k]
        for t in (0, 100, 300, 599 - 1):
            d = poses[t + 1, :3] - poses[t, :3]
            deltas.append(float(np.abs(act[s0 + t, :3] - d / 0.05).max()))
        # gripper label consistency: closing (opening decreasing) -> +1
        d = np.diff(opens)
        lab = act[s0:s0 + len(d), 6]
        close = d < -0.02
        opn = d > 0.02
        if close.any():
            checks.setdefault('grip_close_frac_plus1', []).append(
                float(np.mean(lab[close] > 0)))
        if opn.any():
            checks.setdefault('grip_open_frac_minus1', []).append(
                float(np.mean(lab[opn] < 0)))
    checks['action_vs_pose_delta_max_abs'] = float(max(deltas))
    checks['grip_close_frac_plus1'] = checks.get('grip_close_frac_plus1')
    checks['grip_open_frac_minus1'] = checks.get('grip_open_frac_minus1')
    checks['EXPERT_ACTIONS'] = bool(a['EXPERT_ACTIONS'])
    checks['pass'] = bool(
        checks['ep_start_cumulative'] and checks['proprio_zero_rows'] == 0
        and checks['frames_dat_frame_count'] == len(act)
        and checks['force6_equals_selected_channels']
        and checks['action_vs_pose_delta_max_abs'] < 1e-6
        and checks['max_abs_z_of_column_mean'] < 3.0)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(checks, indent=2))
    print(json.dumps({k: checks[k] for k in (
        'ep_start_cumulative', 'proprio_zero_rows', 'frames_dat_frame_count',
        'force6_equals_selected_channels', 'max_abs_z_of_column_mean',
        'action_vs_pose_delta_max_abs', 'grip_close_frac_plus1',
        'grip_open_frac_minus1', 'pass')}, indent=2))


if __name__ == '__main__':
    main()
