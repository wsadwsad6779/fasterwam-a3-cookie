"""Append the closed-loop B0 vs P2-clamp result section to the root-cause ledger (utf-8 safe)."""
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LEDGER = ROOT / 'ROOTCAUSE_EVIDENCE_LEDGER_20261118.json'

d = json.load(io.open(str(LEDGER), encoding='utf-8'))

d['v6_closed_loop_result'] = {
    'claim': 'Under the reconstructed normalizer the deployed seed17 episode is reproduced bit-exactly; '
             'clamping only the P2 raw-newton history to the corpus per-channel range removes the '
             'squeeze-and-saturate loop (jam 228->0 steps, peak force 446->111 N, plan 95.5->1.27) '
             'but the task still fails, now as a pinch-carry-release sequence that places no cookie.',
    'scope': 'simulator closed loop, single seed 17, same process, two arms; only policy._infer_kwargs '
             'force_history changed in the CL arm; model, weights, normalizer, pm1, Adapter, sim config '
             'and evaluation protocol untouched; no control change deployed outside simulation',
    'identity_gate': {
        'reference': 'forensics_full_wide_out seed17 (600 steps, 60 chunks)',
        'chunks_compared': 60, 'chunks_bit_equal': 60, 'first_divergent_chunk': None,
        'model_out_max_abs_diff': 0.0,
        'measured_rows_compared': 601, 'measured_max_abs_diff': 0.0,
        'gate_passed': True,
        'meaning': 'B0 is the recorded failure trajectory, not merely a similar run; the reconstructed '
                   'normalizer + re-encoded prompt + protocol noise are therefore deterministic and '
                   'the CL arm differs only by the clip',
    },
    'arms': {
        'B0': {'clip': 'none', 'wall_s': 344.54, 'force_absmax_N': 445.82,
               'frames_gt_14.3N': 234, 'frames_gt_50N': 183, 'frames_gt_100N': 141,
               'force_absmax_by_period_N': {'0-100': 25.24, '100-200': 332.83, '200-300': 445.82,
                                            '300-400': 172.90, '400-500': 13.04, '500-600': 10.42},
               'plan_absmax': 95.5, 'plan_absmax_by_10chunks': [1.78125, 43.75, 95.5, 31.25, 15.0, 2.65625],
               'jam_steps_cmd_gt20mm_track_lt0.3': 228, 'first_jam_frame': 147, 'jam_longest_run': 119,
               'gripper_closed_fraction': 0.7188,
               'eef_z_max': 1.8167, 'eef_z_frame_of_max': 595, 'eef_z_end': 1.8142,
               'eef_z_above_1.0_frames': 155,
               'cookies_in_target': 0, 'success': False},
        'CL': {'clip': 'P2 all mask-true rows to per_channel_min/max', 'wall_s': 170.67,
               'force_absmax_N': 111.20,
               'frames_gt_14.3N': 276, 'frames_gt_50N': 42, 'frames_gt_100N': 11,
               'force_absmax_by_period_N': {'0-100': 111.20, '100-200': 102.49, '200-300': 19.82,
                                            '300-400': 59.49, '400-500': 17.45, '500-600': 11.29},
               'plan_absmax': 1.2656, 'plan_absmax_by_10chunks': [1.078, 1.141, 1.156, 1.266, 1.055, 1.172],
               'jam_steps_cmd_gt20mm_track_lt0.3': 0, 'first_jam_frame': None, 'jam_longest_run': 0,
               'gripper_closed_fraction': 0.6406,
               'eef_z_max': 0.92, 'eef_z_frame_of_max': 0, 'eef_z_end': 0.8270,
               'eef_z_above_1.0_frames': 0,
               'cookies_in_target': 0, 'success': False},
    },
    'deltas': {
        'force_absmax_ratio': 0.2495, 'frames_gt_50N_ratio': 0.2295, 'frames_gt_100N_ratio': 0.078,
        'plan_absmax_ratio': 0.01325, 'jam_steps_ratio': 0.0, 'wall_time_ratio': 0.495,
        'first_force_divergence_frame': 12, 'clip_calls': 60, 'clip_calls_with_changes': 57,
    },
    'per_channel_ood_z': {
        'note': 'z = (raw - calib mean)/calib std; corpus support = per_channel_min/max. '
                'Deployment values are the whole-episode extremes of the recorded B0 and the CL arm.',
        'L_wrist_fx': {'corpus_z': [-1.79, 4.98], 'B0_z': [-246.1, 436.5], 'CL_z': [-18.2, 18.3],
                       'B0_frames_outside': 366, 'CL_frames_outside': 100},
        'L_wrist_fy': {'corpus_z': [-5.70, 2.49], 'B0_z': [-107.6, 775.8], 'CL_z': [-170.4, 53.8],
                       'B0_frames_outside': 474, 'CL_frames_outside': 88},
        'L_wrist_fz': {'corpus_z': [-7.43, 2.71], 'B0_z': [-565.8, 2175.7], 'CL_z': [-277.5, 87.6],
                       'B0_frames_outside': 519, 'CL_frames_outside': 183},
        'L_pad_inner': {'corpus_z': [-1.03, 1.23], 'B0_z': [-1.0, 496.3], 'CL_z': [-1.0, 52.2],
                        'B0_frames_outside': 179, 'CL_frames_outside': 286},
        'L_pad_outer': {'corpus_z': [-1.03, 1.18], 'B0_z': [-1.0, 484.5], 'CL_z': [-1.0, 58.2],
                        'B0_frames_outside': 181, 'CL_frames_outside': 291},
        'L_gripper': {'corpus_z': [-1.89, 1.11], 'B0_z': [-7.4, 32.8], 'CL_z': [-2.6, 41.6],
                      'B0_frames_outside': 253, 'CL_frames_outside': 349},
    },
    'free_space_tare': {
        'fy_mean_frames_0_60_N': -10.71,
        'note': 'L_wrist_fy carries a static tare; the whole 42681-row corpus band is [-14.27,-9.45] N, '
                'so the free-space robot already sits at about -5 sigma of that channel. The force '
                'encoder input is OOD before any contact happens.',
    },
    'scene_geometry': {
        'table_top_z_m': 0.750,
        'target_bin_center_xy_m': [0.095, 0.100], 'target_bin_floor_z_m': 0.753,
        'target_bin_rim_z_m': 0.779, 'target_inner_half_size_m': [0.063, 0.035],
        'source_bin_center_xy_m': [0.175, 0.315], 'source_bin_rim_z_m': 0.776,
        'cookie_half_size_m': [0.025, 0.0031667, 0.0125],
        'cookie_row_spacing_m': 0.0088333, 'cookie_gap_m': 0.0025, 'cookie_columns': 4,
        'slots_in_target': 10, 'required_cookies': 10, 'success_hold_steps': 20,
        'note': 'source and target are 0.21 m apart at the same height; the transfer is a short, '
                'low-clearance insertion, so eef z near 0.81 is already a valid carry height.',
    },
    'new_failure_mode_CL': {
        'summary': 'no jam, but no placement: the arm pinches at the source, carries horizontally at '
                   'z~0.81 toward the target footprint, opens, and repeats; cookies_in_target stays 0.',
        'commanded_lift_last_200_frames_mm': {'B0': {'up': 1595.6, 'down': -323.3, 'net_z': 1272.3},
                                              'CL': {'up': 91.8, 'down': -74.5, 'net_z': 17.3}},
        'grasp_segments_opening_lt_0_30': {
            'CL': [
                {'frames': [197, 283], 'opening_floor': 0.175, 'pad_inner_mean_N': 9.24,
                 'pad_outer_mean_N': 9.19, 'gripper_force_mean_N': 18.10,
                 'carry_y_m': [0.222, 0.094], 'eef_z_mean': 0.813,
                 'pad_inner_absmax_N': 16.49, 'pad_outer_absmax_N': 40.59},
                {'frames': [373, 493], 'opening_floor': 0.159, 'pad_inner_mean_N': 11.64,
                 'pad_outer_mean_N': 8.34, 'gripper_force_mean_N': 17.74,
                 'carry_y_m': [0.304, 0.100], 'eef_z_mean': 0.815,
                 'pad_inner_absmax_N': 16.64, 'pad_outer_absmax_N': 15.42},
            ],
            'B0': [
                {'frames': [280, 297], 'opening_floor': 0.294, 'pad_inner_mean_N': 78.71,
                 'pad_outer_mean_N': 179.21, 'gripper_force_mean_N': 21.44, 'eef_z_mean': 0.747},
                {'frames': [333, 371], 'opening_floor': -0.016, 'note': 'fingers fully closed, crushing'},
                {'frames': [479, 600], 'opening_floor': -0.016, 'eef_z': [1.263, 1.814],
                 'note': 'fully closed while climbing to the ceiling'},
            ],
        },
        'squeeze_vs_corpus': 'the entire expert corpus never exceeds 1.54 N on either finger pad or '
                             '2.95 N on the gripper actuator; CL squeezes at 9-12 N mean (40 N peak) '
                             'and B0 at 79-179 N mean, i.e. CL is already 6-12x the corpus pad force',
        'first_contact_CL_f78_121': {
            'per_channel_absmax_N': {'fx': 13.44, 'fy': 111.20, 'fz': 50.24, 'pad_inner': 36.28,
                                     'pad_outer': 35.35, 'gripper': 58.97},
            'cmd_sum_mm': [7.7, -18.6, 44.5], 'disp_sum_mm': [1.2, -3.4, 31.3],
            'grip_cmd_mean': -0.864,
            'read': 'fingers closing while the commanded horizontal motion is blocked and the '
                    'commanded lift is partly absorbed; this is the CL analogue of the B0 jam, '
                    'three orders of magnitude smaller and 60 frames earlier',
        },
        'release_events_CL': {
            'opening_jumps': [[283, 290], [493, 510]],
            'eef_at_release_1': [0.060, 0.094, 0.778],
            'eef_at_release_2': [0.040, 0.100, 0.819],
            'note': 'both releases are inside the target-bin footprint at rim height; no cookie is '
                    'counted, so either nothing was held or the plates were lost before settling',
        },
    },
    'verified': [
        'The reconstructed pipeline reproduces the recorded seed17 600-step episode bit-exactly '
        '(60/60 chunks, 601/601 measured rows), so a single-input counterfactual is well defined.',
        'Unbounded P2 history is a dominant cause of the action saturation and the crash-jam: with '
        'only that input clipped, the plan amplitude returns to the corpus range and jam steps go to 0.',
        'The physical forces remain far outside the corpus support in both arms; clipping the policy '
        'input does not clip the simulator contact.',
        'After clipping, the policy still fails, and fails differently: it never places a cookie.',
    ],
    'unverified': [
        'Whether the CL gripper actually held cookie plates during the two carries; v2 did not log '
        'cookie body positions, so the 0 is unexplained by object state.',
        'Whether the CL failure mode is systematic across seeds or an artefact of the f12 chaos '
        'divergence of this one rollout.',
        'The source of the 6-12x over-squeeze: not established whether it is the force conditioning, '
        'the gripper command head, or the state being off-distribution.',
        'Original normalizer/v1g bytes remain unrecovered (identity marker unchanged).',
    ],
    'artifacts_sha256': {
        'closed_loop_p2clamp_20261118_v2/B0.npz': '1e1af340daf8e023d088a766ef961a58263d153967e3995a643005eda72979ed',
        'closed_loop_p2clamp_20261118_v2/CL.npz': '5f7a55312fcdfd4033026e407bb6f2644f7b703e42fbb36df0c95f250b776025',
        'closed_loop_p2clamp_20261118_v2/analysis_closed_loop.json': 'f7db7dcb34348e99aa360fe203662fcf1e485e18bb457917e4921f70371db837',
        'closed_loop_p2clamp_20261118_v2/analysis_failure_mode.json': 'e91372279a208499f7ba77f5914e219e8e1d969740b2a5d948764a306eb521dc',
        'closed_loop_p2clamp_20261118.py': 'f37d7bf0fe8367bd76e9e361fe237407653d766b46d034fa0270735455479315',
        'analyze_closed_loop.py': 'e466888e91404c8a88bcfc51655b505d749a2957a450de51c968f3f59ad8612c',
        'analyze_closed_loop_failure_mode.py': '26583253626396df7fd5afadd8758a5dbcd885e809ff77aa2541018f1665309b',
    },
    'caveats': [
        'Single seed, single episode, simulator only; no claim of deployment safety or success.',
        'Chaos: a clip of a few newtons at f12 changes the trajectory; the mechanism claim rests on '
        'the structural argument (bounded history cannot re-enter the saturation loop) plus the '
        'aggregate force/jam contrast, not on frame-wise causality after divergence.',
    ],
}

d['next_question'] = ('闭环已证明 P2 裁剪打断挤压正反馈，但 CL 仍 0 块：需记录 cookie 实体位姿，判定两次“夹持搬运”是否真的夹到饼干、'
                      '以及在目标框上方释放后为何未被计数；再判断该失败模式是跨 seed 系统性的还是单条轨迹的混沌结果。')
d['ledger_updated_utc'] = '2026-11-18T11:55:00Z'

io.open(str(LEDGER), 'w', encoding='utf-8').write(json.dumps(d, indent=2, ensure_ascii=False, default=str))
print('ledger keys', len(d))
print('bytes', LEDGER.stat().st_size)