"""Append the minimal-fix and placement sections to the ledger."""
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LEDGER = ROOT / 'ROOTCAUSE_EVIDENCE_LEDGER_20261118.json'
d = json.load(io.open(str(LEDGER), encoding='utf-8'))

d['gripper_fix_closed_loop_20261118'] = {
    'claim': 'A deployment-side temporal filter on the predicted gripper reduces executed chatter by '
             'about half (M5) or a fifth (H8) and removes the one-step open/close pulses, but it does '
             'not put a single plate in the bin and, in this seed, the filtered rollouts reach much '
             'higher contact forces. The chatter is therefore real but not the binding failure.',
    'scope': 'simulator, seed 17, one process per arm, only the predicted dim 7 post-processed inside '
             'the run; model, weights, normalizer, pm1, Adapter and protocol untouched',
    'gate': 'the CL arm reproduces closed_loop_p2clamp_20261118_v2/CL.npz bit-exactly on eef pose, '
            'force and opening (max_abs_diff 0.0)',
    'arms': {
        'CL': {'executed_grip_flips': 77, 'one_step_close_pulses': 20, 'short_close_pulses_le2': 20,
               'close_runs_ge10_followed_by_open': 3, 'force_absmax_N': 111.2,
               'cookies_in_source_final': 69, 'cookies_in_target': 0},
        'M5': {'executed_grip_flips': 37, 'one_step_close_pulses': 1, 'short_close_pulses_le2': 1,
               'close_runs_ge10_followed_by_open': 7, 'force_absmax_N': 1533.2,
               'cookies_in_source_final': 73, 'cookies_in_target': 0},
        'H8': {'executed_grip_flips': 63, 'one_step_close_pulses': 6, 'short_close_pulses_le2': 6,
               'close_runs_ge10_followed_by_open': 8, 'force_absmax_N': 1398.7,
               'cookies_in_source_final': 79, 'cookies_in_target': 0},
    },
    'interpretation': 'flips 77->37 (M5) and 77->63 (H8) with releases preserved or increased, so '
                      'the filter does not suppress real open/close events; the force increase and '
                      'the zero placement are single-seed outcomes and are being repeated on seeds '
                      '11 and 21 before any claim is made',
}

d['cookie_transport_outcome_20261118'] = {
    'claim': 'In every arm the policy does grasp, lift and carry plates toward the bin, but the '
             'plates end against or astride the inner wall, so the strict in-target test (plate '
             'bottom on the bin floor inside the inner footprint) never passes.',
    'geometry': {'target_centre_xy': [0.095, 0.100], 'target_outer_half': [0.063, 0.035],
                 'slot_band': 'x +/-0.028, y +/-0.01267', 'floor_world_z': 0.7575,
                 'resting_plate_centre_z': 0.770, 'rim_z': 0.779},
    'arms': {
        'CL_s17': {'cookies_moved_gt_20mm': 13, 'moved_gt_50mm': 13, 'max_disp_mm': 245,
                   'lifted_gt_20mm': 13, 'final_in_outer_footprint': 3},
        'CL_s11': {'cookies_moved_gt_20mm': 16, 'moved_gt_50mm': 16, 'max_disp_mm': 1073,
                   'lifted_gt_20mm': 8, 'final_in_outer_footprint': 0},
        'CL_s21': {'cookies_moved_gt_20mm': 15, 'moved_gt_50mm': 9, 'max_disp_mm': 183,
                   'lifted_gt_20mm': 9, 'final_in_outer_footprint': 7},
        'M5_s17': {'cookies_moved_gt_20mm': 15, 'moved_gt_50mm': 7, 'max_disp_mm': 215,
                   'lifted_gt_20mm': 7, 'final_in_outer_footprint': 2},
    },
    'plate_positions_that_should_have_counted': {
        'CL_s21': [[0.0555, 0.0668, 0.7682], [0.0552, 0.0738, 0.7682], [0.0550, 0.0805, 0.7682],
                   [0.0561, 0.0882, 0.7711], [0.0558, 0.0999, 0.7748], [0.0707, 0.0991, 0.7871],
                   [0.0703, 0.1159, 0.7882]],
        'note': 'all at resting height (0.768-0.788) but x ~0.055, i.e. 40 mm to the -x side of the '
                'bin centre; a 50 mm plate there spans x 0.030-0.080 while the inner region starts '
                'near x 0.057, so the stack straddles the near wall',
    },
    'expert_release_reference': {
        'n_events': 414,
        'dx_from_centre_mm': {'p10': -30.3, 'p50': -4.6, 'p90': 12.0},
        'dy_from_centre_mm': {'p10': -0.8, 'p50': 97.3, 'p90': 194.7},
        'z': {'p10': 0.7647, 'p50': 0.7656, 'p90': 0.848},
        'note': 'about half of the label-defined releases happen back at the source (return release), '
                'so dy is bimodal; the releases at the bin sit within a few mm of the centre in x',
    },
    'policy_release_positions': {
        'CL_s21': {'release_offsets_mm': [151.9, 27.6, 24.0, 145.3]},
        'M5_s17': {'release_offsets_mm': [135.9, 149.1, 27.7, 163.2, 39.3, 228.6, 225.9]},
        'reading': 'the policy does release near the bin on some events (24-39 mm) but lands the '
                   'stack about 40 mm to the -x side, i.e. at the near wall rather than in the slots',
    },
    'conclusion': 'the binding failure after the force fix is transport/release precision, not the '
                  'gripper chatter: removing the chatter changed nothing about placement',
}

d['next_question'] = ('放置失败的 x 偏置（约 -40 mm）是终点姿态误差还是释放时机误差：需比较模型在 y 到达框区时'
                      '的 eef x 与专家释放点，并把 M5 的多 seed 结果与 CL 对齐；同时继续把 seed11 的'
                      '大位移(1073 mm)与 seed17 的接触力分开追踪。')
d['ledger_updated_utc'] = '2026-11-18T08:20:00Z'

io.open(str(LEDGER), 'w', encoding='utf-8').write(json.dumps(d, indent=2, ensure_ascii=False, default=str))
print('sections', len(d), 'bytes', LEDGER.stat().st_size)