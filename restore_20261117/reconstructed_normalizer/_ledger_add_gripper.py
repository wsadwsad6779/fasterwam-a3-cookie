"""Append the gripper supervision/generation mechanism section to the root-cause ledger."""
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LEDGER = ROOT / 'ROOTCAUSE_EVIDENCE_LEDGER_20261118.json'
d = json.load(io.open(str(LEDGER), encoding='utf-8'))

d['gripper_supervision_and_generation_20261118'] = {
    'claim': 'Action dim 7 is not a command but a label derived from the measured opening, so the '
             'supervision itself toggles sign 3-8 times per 32-frame window at grasp/release '
             'states; the policy reproduces that propensity at matched states and only exceeds it '
             'where it extrapolates. The sampler is not the cause: 50 Euler steps keep the chatter '
             'and the pattern can be bit-identical across noise seeds at some states.',
    'training_data_identity': 'CANDIDATE_RECONSTRUCTED_TRAINING_DATA',
    'training_data_note': 'arrays.npz sha256 b5468c274cc5eede761242478750e13e0a58529dba9b5400ccd2872f7f9fc7f0; '
                          'the calibration file records the original corpus sha256 942164b7... but that '
                          'file was not recovered, so byte identity with the historical training data '
                          'cannot be claimed.',
    'source_chain': {
        'run_config': 'step_040000/run_config.yaml: data.train._target_ = A3VideoDataset, '
                      'dataset_dirs=/workspace/a3_data/a3_fw_v1g, num_frames=33, action 7 / state 14, '
                      'processor.delta_action_dim_mask=[T,T,T,T,T,T,F] (gripper not delta), '
                      'norm min/max, model.loss.lambda_action=1.0, lambda_force=1.0',
        'label_construction': 'finetune/prepare_a3_arrays.py lines 94-107: op=state[:,7]; d=diff(op); '
                              'GRIP_EPS=1e-4; d<-eps -> +1 (closing), d>+eps -> -1 (opening), '
                              '|d|<=eps -> hold previous; g[L-1]=g[L-2]; act[a+L-1]=act[a+L-2]. '
                              'joint_cmd=action[:,7] is stored but NOT used for the label.',
        'temporal_alignment': 'robot_video_dataset.py _get: action is [T-1, 7] and proprio is '
                              'proprio[:-1], so row t pairs the action taken at state t; the label at t '
                              'already uses opening[t+1]-opening[t] (one-step lookahead).',
        'loss': 'fastwam.py training_loss: action_loss_token = mse(pred,target).mean(dim=2) over all 7 '
                'dims equally, masked only by action_is_pad over time; no per-dim weight, no '
                'temporal-difference or consistency term.',
        'sampler': 'fastwam.py infer_action: latents_action = randn(1, horizon=32, 7) (iid across rows '
                   'and dims), 10-step schedule, shift 5; scheduler_continuous.py step is plain Euler; '
                   'the last step covers sigma 0.357 -> 0.',
        'action_expert': 'sparse_action_dit.py: action_encoder = Linear(7, hidden), RoPE freqs 0..31, '
                         'head = Linear(hidden, 7); no per-dimension special casing anywhere.',
    },
    'corpus_statistics': {
        'rows': 42681, 'episodes': 100, 'ep_len_mean': 426.8,
        'label_plus1_fraction': 0.6621,
        'rule_reproduction_mismatch': '0 / 42681',
        'flips_per_episode': {'mean': 18.92, 'median': 19, 'min': 15, 'max': 23},
        'label_run_length_frames': {'p50': 5, 'p90': 73.8, 'p99': 140},
        'flip_d_opening': {'p50': 0.000549, 'p90': 0.0628, 'frac_lt_1e-3': 0.5687,
                           'frac_lt_1e-2': 0.7828},
        'windows_32': {'n': 39481, 'mean_flips': 1.2135, 'p_any': 0.4817, 'p_constant': 0.5183,
                       'classes': {'constant': 0.5183, 'single_step': 0.2337,
                                   'multi_step': 0.2213, 'alternating': 0.0267},
                       'max_flips': 10, 'p99_flips': 8, 'frac_ge_6': 0.0537},
        'window_10': {'mean_flips_of_9': 0.3909, 'p_any': 0.2110},
    },
    'generation_statistics_seed17_60chunks': {
        'mean_flips_32': 3.7333, 'mean_flips_first10': 1.25,
        'p_constant_32': 0.55,
        'classes': {'constant': 0.55, 'single_step': 0.05, 'multi_step': 0.35, 'alternating': 0.05},
        'replan_boundary_flip_rate': 0.3559, 'within_chunk_row_flip_rate': '0.083-0.200',
        'flip_decision_margin': {'n': 224, 'p50': 0.359, 'frac_lt_0.2': 0.411,
                                 'frac_gt_0.8': 0.371},
        'position_structure': {'mod4_mean_spread': 0.129, 'global_std': 23.12,
                               'mod4_boundary_flip_rate': 0.138, 'mod4_elsewhere': 0.115,
                               'conclusion': 'no per-4-row grouping or RoPE index artefact'},
    },
    'state_sweep_9states_5seeds': {
        'note': 'offline single-action inference at recorded replan states; protocol seed plus '
                '1001-1004; flip counts are per 32-row plan',
        'rows': [
            {'frame': 0, 'model_flips': [7, 5, 5, 5, 7], 'knn20_label_flips': 4.75,
             'label_const_frac': 0.0, 'nn_dist': 0.0, 'eef_z': 0.920, 'force_sigma': 1.0,
             'opening': 1.000},
            {'frame': 30, 'model_flips': [3, 3, 3, 3, 3], 'knn20_label_flips': 3.05,
             'label_const_frac': 0.0, 'nn_dist': 0.32, 'eef_z': 0.838, 'force_sigma': 1.0,
             'opening': 0.852},
            {'frame': 50, 'model_flips': [4, 10, 5, 1, 4], 'knn20_label_flips': 3.55,
             'label_const_frac': 0.0, 'nn_dist': 1.54, 'eef_z': 0.794, 'force_sigma': 1.0,
             'opening': 1.000},
            {'frame': 80, 'model_flips': [0, 0, 0, 1, 0], 'knn20_label_flips': 0.0,
             'label_const_frac': 1.0, 'nn_dist': 0.75, 'eef_z': 0.813, 'force_sigma': 3.7,
             'opening': 0.187},
            {'frame': 110, 'model_flips': [6, 4, 4, 5, 2], 'knn20_label_flips': 3.25,
             'label_const_frac': 0.0, 'nn_dist': 0.61, 'eef_z': 0.826, 'force_sigma': 4.3,
             'opening': 1.002},
            {'frame': 140, 'model_flips': [0, 0, 0, 0, 0], 'knn20_label_flips': 0.25,
             'label_const_frac': 0.75, 'nn_dist': 0.10, 'eef_z': 0.763, 'force_sigma': 55.2,
             'opening': 0.451},
            {'frame': 200, 'model_flips': [0, 0, 0, 0, 0], 'knn20_label_flips': 0.0,
             'label_const_frac': 1.0, 'nn_dist': 0.79, 'eef_z': 0.740, 'force_sigma': 290.5,
             'opening': 0.468},
            {'frame': 300, 'model_flips': [0, 0, 0, 2, 0], 'knn20_label_flips': 0.0,
             'label_const_frac': 1.0, 'nn_dist': 6.46, 'eef_z': 0.785, 'force_sigma': 280.0,
             'opening': 0.303},
            {'frame': 440, 'model_flips': [10, 12, 17, 19, 13], 'knn20_label_flips': 7.25,
             'label_const_frac': 0.0, 'nn_dist': 4.79, 'eef_z': 0.964, 'force_sigma': 17.5,
             'opening': 0.522},
        ],
        'reading': 'the model tracks the matched-state label flip propensity within 1.3x at '
                   'in-distribution states (f0 5.8 vs 4.75, f30 3.0 vs 3.05, f110 4.2 vs 3.25), is '
                   'quiet where the labels are constant (f80/f140/f200/f300), and exceeds the '
                   'labels 2x only at f440, the post-jam pose at the target bin (nn distance 4.8)',
        'corr_flips_vs_eef_z': 0.879, 'corr_flips_vs_force_sigma': -0.444,
        'note_on_correlation': 'the z correlation is carried almost entirely by the single f440 '
                               'point; f200 sits at 290 sigma with zero chatter, so chatter does '
                               'not track OOD by itself',
    },
    'gpu_controls': {
        'gate': 'f440 and f80 protocol-noise jobs reproduce recorded chunks 44 and 8 bit-exactly '
                '(max_abs_diff 0.0)',
        'steps_50_same_noise': {'f440': {'flips_10steps': 10, 'flips_50steps': 14,
                                         'sign10_equal': True},
                                'f80': {'flips_10steps': 0, 'flips_50steps': 0,
                                        'sign10_equal': True}},
        'seed_spread_f440_10steps': [12, 17, 19, 13],
        'seed_spread_f440_sign10': ['+-+++++--+', '+---++-+--', '+---++-+--', '+--+-----+'],
        'per_row_agreement_with_protocol_seed': 0.5,
        'verdict': 'M1 (Euler discretisation) rejected; the per-seed sign pattern changes but every '
                   'seed stays chattery, so the chatter is a property of the conditional '
                   'distribution at that state, not of one noise draw',
    },
    'direct_exhibit_f50_nearest_demo_window': {
        'opening_head': [0.999810, 1.000160, 0.999750, 0.910020, 0.825200, 0.747490],
        'd_opening_head': [0.000354, -0.000410, -0.089737, -0.084812, -0.077718, -0.080185],
        'label_head': '-+++++...',
        'joint_cmd_head': [1.0, 1.0, 0.92, 0.84, 0.76, 0.68],
        'read': 'the first two frames differ by 4e-4 in opening (0.04% of the aperture) and already '
                'flip the label; after the close completes, settling jitter of 1e-4..1e-2 produces '
                'four more flips while joint_cmd stays constant at 0.4486',
    },
    'conclusions': {
        'root_of_gripper_chatter': 'label construction: hard sign of the measured-opening first '
                                   'difference with a 1e-4 deadband, no hysteresis, one-step '
                                   'lookahead; 57% of all label flips come from |d opening| < 1e-3 '
                                   'and the expert command itself is smooth',
        'policy_behaviour': 'the model reproduces the label flip propensity at matched states, and '
                            'adds a factor ~2 only where the pose leaves the demo support (f440, '
                            'nn distance 4.8)',
        'not_the_cause': 'sampler discretisation (50 steps), per-row noise (deterministic at some '
                         'states), action-token indexing or grouping (no mod-4 structure), and a '
                         'gripper-specific code path (dim 7 uses the same DiT and MSE as dims 0-5)',
        'open': 'whether the executed chatter (a 50 ms open pulse inside a grasp) is what makes the '
                'plates leave the source bin yet never settle in the target; the closed-loop fix '
                'test is running',
    },
    'post_process_offline': {
        'median_k5': {'mean_flips_32': 1.1333, 'mean_flips_first10': 0.3167,
                      'executed_majority_intent_changes': 2},
        'hysteresis': {'h0.2': 2.5167, 'h0.8': 1.5833, 'h0.95': 1.4667},
        'note': 'median k=5 brings the model back to the supervision rate; hysteresis alone cannot '
                'remove confident alternation (37% of flips have a margin above 0.8)',
    },
    'artifacts': {
        'GRIPPER_SUPERVISION_VS_GENERATION.json': 'corpus vs model statistics',
        'GRIPPER_MATCHED_PHASE.json': 'pattern classes, phase matching, per-chunk listing',
        'GRIPPER_MARGIN_HYSTERESIS.json': 'flip margins and hysteresis sweep',
        'GRIPPER_STRUCTURE.json': 'mod-4/mod-8 structure and median filter',
        'GRIPPER_STATE_MATCHED.json': 'nearest-neighbour and kNN-20 label comparison',
        'gpu_gripper_mechanism_20261118/': 'steps-50 and noise-seed controls with bit-exact gates',
        'gpu_gripper_state_sweep_20261118/': '9 states x 5 seeds sweep',
    },
}

d['next_question'] = ('夹爪抖动的数据层机制已定位（标签=实测开度差分符号）。待闭环修复实验回答：'
                      '时间中值/滞回能否在不压制必要开合的前提下减少执行层抖动，并改善 cookies_in_target；'
                      '同时把 seed11 的姿态漂移与 seed17 的接触力作为独立机制继续分开追踪。')
d['ledger_updated_utc'] = '2026-11-18T07:45:00Z'

io.open(str(LEDGER), 'w', encoding='utf-8').write(json.dumps(d, indent=2, ensure_ascii=False, default=str))
print('sections', len(d), 'bytes', LEDGER.stat().st_size)