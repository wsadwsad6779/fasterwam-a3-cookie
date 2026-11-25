"""P13: F0 stage-1 gripper-mode mechanism analysis.

Reads the F0 training probe series (verbatim from fwv1/logs/f0_v3_baseline.log)
and the F2 milestone gripper series (p_items/T7_GRASP_ROOT_CAUSE_15K.json),
then writes a falsifiable comparison artifact. Analysis only; does not change
any training or evaluation state.
"""
import json
import os
import time

P = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TS = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())

f0_probes = [
    {'step': 10, 'lt_0.30': 0.0, 'median': 0.7641, 'open_ge_0.8': 0.4531},
    {'step': 500, 'lt_0.30': 0.0, 'median': 0.3433, 'open_ge_0.8': 0.0},
    {'step': 1000, 'lt_0.30': 0.0, 'median': 0.4136, 'open_ge_0.8': 0.0},
    {'step': 1500, 'lt_0.30': 1.0, 'median': 0.2364, 'open_ge_0.8': 0.0},
    {'step': 2000, 'lt_0.30': 0.875, 'median': 0.2687, 'open_ge_0.8': 0.0},
    {'step': 2500, 'lt_0.30': 0.296875, 'median': 0.3137, 'open_ge_0.8': 0.0},
    {'step': 3000, 'lt_0.30': 0.625, 'median': 0.29125, 'open_ge_0.8': 0.0},
]
late = [p for p in f0_probes if p['step'] >= 1500]
closed_mean = sum(p['lt_0.30'] for p in late) / len(late)
open_mean = sum(p['open_ge_0.8'] for p in late) / len(late)
median_mean = sum(p['median'] for p in late) / len(late)

f2 = json.load(open(os.path.join(P, 'T7_GRASP_ROOT_CAUSE_15K.json')))
f2_lt = f2['verified']['executed_grasp_level_fraction']
f2_med = f2['verified']['executed_opening_command_median']

artifact = {
    'schema': 'task7_mechanism_analysis_v1',
    'id': 'P13_F0_STAGE1_GRIP_MECHANISM',
    'created_utc': TS,
    'status': 'stage1 probe analysis; official F0 3k 10-seed eval pending',
    'subject': 'Does the F0 no-force baseline reproduce the F2 0.43 mid-band collapse?',
    'data_sources': {
        'f0_probe_log': '/workspace/fwv1/logs/f0_v3_baseline.log (grip_probe field every 500 steps; 2 val samples, 32-step infer)',
        'f0_checkpoint': '/workspace/fwv1/f0_v3_baseline/step_003000.pt (3,354,213,309 B)',
        'f2_gripper_series': 'p_items/T7_GRASP_ROOT_CAUSE_15K.json#verified',
        'f2_3k_eval': 'p_items/SUMMARY_step_3000.json (seeds 7100-7109)',
        'f2_15k_eval': 'p_items/SUMMARY_step_15000.json (seeds 7500-7509)'},
    'f0_probe_series': f0_probes,
    'f0_derived': {
        'early_10_1000': {'median': [0.7641, 0.3433, 0.4136], 'lt_0.30': [0.0, 0.0, 0.0],
                          'open_ge_0.8': [0.4531, 0.0, 0.0],
                          'reading': 'early mid-band; open mode present only at step 10'},
        'late_1500_3000': {'closed_mean': closed_mean, 'open_mean': open_mean,
                           'median_mean': median_mean,
                           'median_range': [min(p['median'] for p in late), max(p['median'] for p in late)],
                           'reading': 'closed-biased single band; open mode absent in all late probes'},
        'closed_minus_open_gap': closed_mean - open_mean},
    'f2_reference': {
        'exec_lt_030_series': f2_lt,
        'exec_median_series': f2_med,
        'f2_3k_eval': {'in_target_mean': 0.0, 'exact_fill_rate': 0.0, 'grasps_mean': 11.6,
                       'open_cmd_mean': 0.0175, 'close_cmd_mean': 0.97,
                       'force_max_mean_n': 105.25,
                       'checkpoint_sha256': '342c4c4a74e63598a6a565eda59db2570878d842742fc65ba7ad343db058a34e'},
        'f2_15k_eval': {'in_target_mean': 0.0, 'exact_fill_rate': 0.0,
                        'open_cmd_mean': 0.02917, 'close_cmd_mean': 0.9388,
                        'force_max_mean_n': 25.39,
                        'checkpoint_sha256': '83aacf23cc146af4b33027ab1cfaea0ce8fd0a0aac6c28cf683fdfc69dbab2e1'}},
    'comparison': {
        'step_matched_3k': {
            'f0_probe_median': 0.29125, 'f2_3k_exec_median': f2_med['3k'],
            'f0_probe_lt_030': 0.625, 'f2_3k_exec_lt_030': f2_lt['3k'],
            'reading': 'at the 3k boundary F0 is closed-biased (median 0.29, probe <0.30 62.5%) while F2 sits at 0.39 with only 1.05% <0.30'},
        'caveat': 'F0 probe = 2 fixed val samples x 32-step predictions; F2 numbers = executed closed-loop commands over 10 seeds. Direction is clear but the official F0 3k eval is the arbiter.'},
    'hypotheses_ranked': [
        {'id': 'H1_force_bundle_biases_midband', 'status': 'supported_but_not_proven',
         'claim': 'the force bundle (FiLM + force history + aux force head/loss + dropout) is associated with the release-band 0.43 bias; without it F0 goes closed',
         'evidence': 'step-matched 3k: F0 probe closed 62.5% vs F2 1.05% <0.30; F2 series stays 0.39-0.45 from 3k to 15k while F0 drops to 0.24-0.31',
         'weakness': 'probe vs eval protocol; no 6k-15k F0 points yet'},
        {'id': 'H2_shared_training_degeneracy', 'status': 'not_exonerated',
         'claim': 'shared continuous V3 + flow-MSE training can produce degenerate modes; F0 closed-only is another degenerate mode',
         'evidence': 'F0 open_ge_0.8 = 0 in every late probe; F2 open_cmd is only 1.75%-2.9%; neither model learns a healthy open/close bimodal policy',
         'weakness': 'open may be rare in the val observations; needs per-phase eval'},
        {'id': 'H3_open_mode_failure_is_separate', 'status': 'plausible',
         'claim': 'the approach/open failure is a separate axis from the close-band bias; it appears in both F0 and F2',
         'evidence': 'F0 late open 0%; F2 3k open 1.75%, 15k open 2.92%',
         'weakness': 'probe uses fixed val observations; the rollout may open more'}],
    'falsification_plan': [
        'official F0 3k 10-seed eval (seeds 7100-7109): executed open_cmd_frac, <0.30 fraction, median, in_target',
        'F0 6k/9k/12k/15k probes + official evals: if F0 drifts to 0.43 mid-band by 15k, H1 loses; if it stays closed/bimodal, H1 gains',
        'F2 3k paired eval (same seeds, frozen sim) to make the 3k contrast like-for-like'],
    'conclusion': ('At stage1 the no-force F0 does not reproduce the F2 0.43 mid-band: it becomes closed-biased '
                   '(late-probe median 0.29, <0.30 62.5% at 3k, open 0%). F2 at 3k is still at 0.39 with 1.05% <0.30 and only '
                   'moves to 0.43 later. This is evidence that the force bundle is associated with the release-band bias, '
                   'but it is not proof: the F0 probe is small, the official 3k eval is pending, and the decisive test is '
                   'the F0 6k-15k trajectory. No causal claim yet.'),
    'actions': {'training': 'F0 continuation 3k->15k per user 2026-11-25T04:03:50Z (unconditional)',
                'eval': 'F0 3k official eval running; F2 3k paired next; r sweep queued'},
    'paths': {'f0_run': '/workspace/fwv1/f0_v3_baseline',
              'f0_eval_3k': '/workspace/fwv1/eval_fingertip_v1/cmp_f0_3k',
              'f2_eval_3k': '/workspace/fwv1/eval_fingertip_v1/cmp_f2_3k'}}

out = os.path.join(P, 'P13_F0_STAGE1_GRIP_MECHANISM.json')
json.dump(artifact, open(out, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
print('WROTE', out, os.path.getsize(out), 'closed_mean=%.3f open_mean=%.3f median_mean=%.3f'
      % (closed_mean, open_mean, median_mean))