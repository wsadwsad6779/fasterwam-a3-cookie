"""Read-only local CPU audit. Run: py -3.13 <this file>. No torch/network imports."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
import ast
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

BASE = Path(__file__).resolve().parent
REPO = BASE.parents[1]
SOURCE = BASE / 'forensics_download/training_source_a3_fw_force_arrays.npz'
UPSTREAM = BASE / 'forensics_download/training_source_a3_fw_arrays.npz'
PREP = REPO / 'finetune/prepare_a3_arrays.py'
DATASET = REPO / 'stage_40k/fw/src/fasterwam/datasets/lerobot/a3_video_dataset.py'
CONFIG = BASE.parent / 'step_040000/run_config.yaml'
CALIB = BASE.parent / 'step_040000/force_calib_v1.json'
OUT = BASE / 'TRAINING_LABEL_MECHANISM_20261118.json'
EXPECTED = 'd130363f13ea511fdbad277e9d1ddcc7c4c9712f0661c898bada2b34e7aac7dd'

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()

def summary(a):
    a = np.asarray(a)
    return {'n': int(a.size), 'min': float(a.min()), 'median': float(np.median(a)),
            'p90': float(np.percentile(a, 90)), 'p99': float(np.percentile(a, 99)),
            'max': float(a.max())} if a.size else {'n': 0}

def flips(g):
    return np.flatnonzero(np.diff(g) != 0) + 1

def labels(op, eps=1e-4, hysteresis=False):
    # Stored opening is float32; original preprocessing used pre-storage float64.
    d = np.diff(op.astype(np.float64))
    g = np.full(len(op), -1, dtype=np.int8)
    extreme = float(op[0])
    state = -1
    for t, delta in enumerate(d):
        if hysteresis:
            v = float(op[t+1])
            extreme = max(extreme, v) if state == -1 else min(extreme, v)
            if (state == -1 and extreme-v > eps) or (state == 1 and v-extreme > eps):
                state = -state
                extreme = v
        elif delta < -eps:
            state = 1
        elif delta > eps:
            state = -1
        g[t] = state
    if len(g) > 1:
        g[-1] = g[-2]
    return g

def hist(x):
    k, v = np.unique(x, return_counts=True)
    return {str(int(a)): int(b) for a, b in zip(k, v)}

def window_stats(g):
    # 33 state/image frames, 32 action tokens, no padding, all starts stride 1.
    counts = {}
    for horizon in (10, 32):
        f = (np.diff(g) != 0).astype(np.int64)
        cs = np.r_[0, np.cumsum(f)]
        starts = np.arange(len(g)-32)
        vals = cs[starts+horizon-1] - cs[starts]
        counts[str(horizon)] = vals
    return counts

def macro_events(cmd):
    d = np.diff(cmd.astype(np.float64))
    active = np.flatnonzero(np.abs(d) > 1e-4)
    groups = []
    for t in active:
        direction = 1 if d[t] < 0 else -1
        if not groups or groups[-1]['direction'] != direction:
            groups.append({'t': int(t), 'end': int(t+1), 'direction': direction, 'travel': abs(float(d[t]))})
        else:
            groups[-1]['end'] = int(t+1)
            groups[-1]['travel'] += abs(float(d[t]))
    return [e for e in groups if e['travel'] >= .1]

def main():
    assert sha(SOURCE) == EXPECTED, 'force asset SHA mismatch'
    assert sha(UPSTREAM) == 'b5468c274cc5eede761242478750e13e0a58529dba9b5400ccd2872f7f9fc7f0'
    z = np.load(SOURCE, allow_pickle=False)
    u = np.load(UPSTREAM, allow_pickle=False)
    prep_text = PREP.read_text(encoding='utf-8')
    ds_text = DATASET.read_text(encoding='utf-8')
    cfg = CONFIG.read_text(encoding='utf-8')
    assert 'self._windows.extend(range(int(start), int(start) + n))' in ds_text
    assert 'n = int(length) - self.num_frames + 1' in ds_text
    assert 'self._action[s][:-1]' in ds_text
    assert 'num_frames: 33' in cfg and 'global_sample_stride: 1' in cfg
    eps_node = next(n for n in ast.parse(prep_text).body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'GRIP_EPS' for t in n.targets))
    assert ast.literal_eval(eps_node.value) == 1e-4
    calib = json.loads(CALIB.read_text(encoding='utf-8'))
    assert EXPECTED in json.dumps(calib)
    specs = [('deadzone_0.0001', .0001, False), ('deadzone_0.0003', .0003, False),
             ('deadzone_0.001', .001, False), ('deadzone_0.003', .003, False),
             ('hysteresis_0.001', .001, True), ('hysteresis_0.003', .003, True),
             ('hysteresis_0.01', .01, True)]
    names = ['stored'] + [s[0] for s in specs]
    acc = {k: {'flips': 0, 'changed_frames': 0, 'windows10': [], 'windows32': [],
               'macro_correct_at_onset': 0, 'macro_missed_within20': 0,
               'macro_label_transition_missing_pm20': 0, 'macro_transition_shift_frames': [],
               'macro_first_correct_delay_frames': []} for k in names}
    episodes, windows, initial = [], [], []
    total = len(z['opening'])
    assert np.array_equal(z['ep_start'], np.r_[0, np.cumsum(z['ep_len'])[:-1]])
    assert int(z['ep_len'].sum()) == total
    command_events = 0
    for ei, (start, length, ep_id) in enumerate(zip(z['ep_start'], z['ep_len'], z['ep_ids'])):
        a, L = int(start), int(length)
        b = a+L
        op, cmd, stored = z['opening'][a:b], z['joint_cmd'][a:b], z['action'][a:b, 6]
        frame, time = z['frame_index'][a:b], z['timestamp'][a:b]
        assert L >= 33
        candidates = {'stored': stored, **{name: labels(op, ep, hy) for name, ep, hy in specs}}
        reconstructed = candidates['deadzone_0.0001']
        mismatch = np.flatnonzero(reconstructed != stored)
        delta = np.diff(op.astype(np.float64))
        events = macro_events(cmd)
        command_events += len(events)
        row = {'episode_position': ei, 'episode_id': int(ep_id), 'start': a, 'length': L,
               'valid_windows': L-32, 'frame_first_last': [int(frame[0]), int(frame[-1])],
               'frame_nonunit_count': int(np.sum(np.diff(frame) != 1)),
               'timestamp_nonpositive_count': int(np.sum(np.diff(time) <= 0)),
               'timestamp_step': summary(np.diff(time)),
               'timestamp_frame_over20_max_error': float(np.max(np.abs(time-frame/20))),
               'episode_index_mismatch': int(np.sum(z['episode_index'][a:b] != ep_id)),
               'label_mismatch_count': len(mismatch), 'label_mismatch_local_frames': mismatch.tolist(),
               'stored_flips': len(flips(stored)), 'deadzone_delta_count': int(np.sum(abs(delta) <= 1e-4)),
               'near_threshold_1e_minus7_count': int(np.sum(abs(abs(delta)-1e-4) <= 1e-7)),
               'initial_label': float(stored[0]), 'initial_delta': float(delta[0]),
               'macro_command_events': events}
        episodes.append(row)
        wc = window_stats(stored)
        for s in range(L-32):
            windows.append((int(wc['32'][s]), ei, s, int(wc['10'][s])))
        initial.append({'episode_id': int(ep_id), 'flips10': int(wc['10'][0]), 'flips32': int(wc['32'][0]),
                        'opening_net32': float(op[32]-op[0]), 'command_net32': float(cmd[32]-cmd[0])})
        for name, g in candidates.items():
            st = acc[name]
            st['flips'] += len(flips(g))
            st['changed_frames'] += int(np.sum(g != stored))
            counts = window_stats(g)
            st['windows10'].extend(counts['10'].tolist())
            st['windows32'].extend(counts['32'].tolist())
            f = flips(g)
            for e in events:
                t, direction = e['t'], e['direction']
                st['macro_correct_at_onset'] += int(g[t] == direction)
                end = min(t+21, e['end']+1, L)
                hit = np.flatnonzero(g[t:end] == direction)
                if len(hit):
                    st['macro_first_correct_delay_frames'].append(int(hit[0]))
                else:
                    st['macro_missed_within20'] += 1
                near = f[(abs(f-t) <= 20) & (g[f] == direction)]
                if len(near):
                    best = int(near[np.argmin(abs(near-t))])
                    st['macro_transition_shift_frames'].append(best-t)
                else:
                    st['macro_label_transition_missing_pm20'] += 1
    def detail(item):
        f32, ei, s, f10 = item
        a = int(z['ep_start'][ei])+s
        op = z['opening'][a:a+33].astype(np.float64)
        cmd = z['joint_cmd'][a:a+33].astype(np.float64)
        g = z['action'][a:a+32, 6]
        return {'evidence': 'E1_DIRECT_STORED_FLOAT32', 'episode_id': int(z['ep_ids'][ei]),
                'local_start': s, 'global_start': a, 'flips10': f10, 'flips32': f32,
                'opening33': op.tolist(), 'opening_delta32': np.diff(op).tolist(),
                'opening_net32': float(op[-1]-op[0]), 'opening_range33': float(np.ptp(op)),
                'opening_total_variation32': float(abs(np.diff(op)).sum()),
                'joint_cmd33': cmd.tolist(), 'joint_cmd_delta32': np.diff(cmd).tolist(),
                'command_net32': float(cmd[-1]-cmd[0]), 'command_range33': float(np.ptp(cmd)),
                'labels32': g.tolist(), 'label_flip_offsets': flips(g).tolist(),
                'counterfactual_labels32': {name: labels(z['opening'][int(z['ep_start'][ei]):int(z['ep_start'][ei]+z['ep_len'][ei])], eps, hy)[s:s+32].tolist() for name, eps, hy in specs}}
    selected, used = [], set()
    for item in sorted(windows, reverse=True):
        if all(item[1] != e or abs(item[2]-s) >= 33 for e, s in used):
            selected.append(detail(item)); used.add((item[1], item[2]))
        if len(selected) == 6:
            break
    # Additional explicit initial and constant-command dense windows.
    initial_details = [detail(next(w for w in windows if w[1] == ei and w[2] == 0)) for ei in range(3)]
    constant_dense = []
    for item in sorted(windows, reverse=True):
        a = int(z['ep_start'][item[1]])+item[2]
        if item[0] >= 2 and np.ptp(z['joint_cmd'][a:a+33]) <= 1e-6:
            constant_dense.append(detail(item))
            if len(constant_dense) == 2:
                break
    baseline = acc['stored']['flips']
    for name, st in acc.items():
        st['flip_reduction_fraction'] = 1-st['flips']/baseline
        st['frame_disagreement_fraction'] = st['changed_frames']/total
        st['segments_total'] = st['flips']+len(episodes)
        for h in (10, 32):
            values = st.pop('windows'+str(h))
            st['window'+str(h)] = {'histogram': hist(values), 'summary': summary(values),
                                     'at_least_2': int(np.sum(np.asarray(values) >= 2))}
        for field in ('macro_transition_shift_frames', 'macro_first_correct_delay_frames'):
            vals = st[field]
            st[field+'_summary'] = summary(vals)
            st[field+'_histogram'] = hist(vals)
            del st[field]
    upstream_compare = {k: {'same_shape': z[k].shape == u[k].shape,
                            'equal': bool(np.array_equal(z[k], u[k]))} for k in u.files if k in z.files}
    result = {'NORMALIZER_IDENTITY': 'RECONSTRUCTED_DETERMINISTIC_NOT_ORIGINAL',
        'original_stats_sha256': 'UNKNOWN',
        'schema': 'training_label_mechanism_cpu_v1', 'python': sys.version,
        'execution': 'CPU NumPy only; no network, GPU, model inference, training or source writes',
        'normalizer': {'status': 'RECONSTRUCTED_NOT_ORIGINAL', 'used_in_this_audit': False,
                       'original_stats_sha256': 'UNKNOWN', 'scope': 'raw stored source-copy labels; no normalizer causality claim'},
        'evidence_levels': {'E1': 'direct local bytes / arithmetic', 'E2': 'local source-code/config/calibration linkage',
                            'E3': 'offline counterfactual or operational proxy, not behavioral causality'},
        'identity': {'force_source_path': str(SOURCE), 'force_source_sha256': sha(SOURCE),
                     'calibration_force_binding_verified': True, 'calibration': calib,
                     'upstream_path': str(UPSTREAM), 'upstream_sha256': sha(UPSTREAM),
                     'upstream_comparison': upstream_compare,
                     'critical_limit': 'Force-source embedded action copy is NOT proof of v1g actual 40k action labels. Dataset reads actions from dataset_dirs v1g, force18 from force_arrays_path; only ep_start/ep_len/row count checked. Upstream a3_fw is 8D proprio, not v1g942 identity.'},
        'source_files': {str(p): sha(p) for p in (PREP, DATASET, CONFIG, CALIB, Path(__file__))},
        'window_rule': {'evidence': 'E2', 'state_frames': 33, 'action_tokens': 32, 'stride': 1,
                         'valid_starts_per_episode': '0..L-33 inclusive; count L-32; no episode crossing/padding',
                         'first10': 'internal transitions among tokens 0..9 (9 opportunities)',
                         'full32': 'internal transitions among tokens 0..31 (31 opportunities)',
                         'opening33_used': 'label at t uses opening[t+1]-opening[t]', 'windows_total': len(windows)},
        'mechanism': {'evidence': 'E2+E1', 'closing_label': 1, 'opening_label': -1, 'deadzone': 'abs(delta)<=1e-4 inherits; strict >/<; first deadzone defaults -1; last label duplicates penultimate',
                      'float_precision_limit': 'original op float64 lost; stored opening float32. Mismatch near threshold cannot alone falsify original implementation.',
                      'mismatches_total': sum(e['label_mismatch_count'] for e in episodes),
                      'all_episode_first_labels': hist([e['initial_label'] for e in episodes])},
        'continuity': {'episodes': len(episodes), 'rows': total, 'ids_strictly_increasing': bool(np.all(np.diff(z['ep_ids']) > 0)),
                        'frame_nonunit_total': sum(e['frame_nonunit_count'] for e in episodes),
                        'timestamp_nonpositive_total': sum(e['timestamp_nonpositive_count'] for e in episodes),
                        'episode_index_mismatch_total': sum(e['episode_index_mismatch'] for e in episodes)},
        'macro_proxy': {'evidence': 'E3', 'events': command_events,
                         'definition': 'joint_cmd deltas >1e-4, group consecutive active same-sign movements (holds ignored), cumulative command travel >=0.1; start at forward delta index',
                         'units': 'opening/joint_cmd native units, not calibrated physical aperture',
                         'timing': 'first correct label within min(20 frames, command-run end); signed nearest same-direction transition within +/-20; matches not one-to-one; 20Hz => 50ms/frame',
                         'warning': 'proxy depends on 0.1 cutoff; not task success, optimal control or definitive ground truth'},
        'offline_candidates': {'evidence': 'E3', 'reset_per_episode': True, 'deadzone': 'forward delta threshold plus inherited state',
                               'hysteresis': 'switch after cumulative retreat from running directional extreme > threshold; causal after next measurement, no future beyond baseline delta; no min dwell',
                               'falsification': 'reject candidate if macro missed events or timing delays unacceptable; jointly inspect dense windows and true command reversals, never pick on flip count alone',
                               'source_data_unchanged': True, 'comparison': acc},
        'initial_windows': initial, 'initial_examples': initial_details,
        'dense_examples': selected, 'constant_command_dense_examples': constant_dense,
        'true_command_reversal_examples': [],
        'episodes': episodes,
        'interpretation_limits': ['Tiny measured opening motion with constant command supports measured micro-motion, not proof of sensor noise (no independent sensor/noise calibration).',
             'Opening movement and same-direction command changes support commanded motion, not every binary label flip being a real command reversal.',
             'Threshold crossing maps small continuous variation to full +/-1 targets (jump 2), then deadzone inheritance prolongs it: deterministic preprocessing amplification, not evidence normalizer created it.',
             'This recovered force asset corrects a missing local identity search result; it does not recover v1g actions, original normalizer, or prove inference failure causality.']}
    for ei in range(2):
        events = episodes[ei]['macro_command_events']
        for previous, event in zip(events, events[1:]):
            if previous['direction'] != event['direction']:
                s = max(0, min(event['t']-8, episodes[ei]['length']-33))
                item = next(w for w in windows if w[1] == ei and w[2] == s)
                example = detail(item)
                example['previous_command_event'] = previous
                example['current_command_event'] = event
                result['true_command_reversal_examples'].append(example)
    result['initial_window_histograms'] = {str(h): hist([w['flips'+str(h)] for w in initial]) for h in (10, 32)}
    result['continuity']['timestamp_frame_over20_max_error'] = max(e['timestamp_frame_over20_max_error'] for e in episodes)
    assert sha(SOURCE) == EXPECTED
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'output': str(OUT), 'continuity': result['continuity'], 'mismatches': result['mechanism']['mismatches_total'],
                      'windows': len(windows), 'macro_events': command_events, 'comparison': acc,
                      'examples': [{k: e[k] for k in ('episode_id','local_start','flips10','flips32','opening_net32','opening_range33','command_range33')} for e in selected+constant_dense]}, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
