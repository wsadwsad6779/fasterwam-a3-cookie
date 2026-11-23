import os, sys, json, time, traceback

os.environ.setdefault('MUJOCO_GL', 'egl')
sys.path.insert(0, '/workspace/fwv1/sim_repo/src')
sys.path.insert(0, '/workspace')

import numpy as np
import pyarrow.parquet as pq

from t7_fingertip_processor import FingertipForceProcessor, CHANNELS

PARQ = ['/workspace/fwv1/p_items/hf_multi/data/chunk-000/file-000.parquet',
        '/workspace/fwv1/p_items/hf_multi/data/chunk-000/file-001.parquet']
SPLIT = '/workspace/fwv1/p_items/DATASET_800_SPLIT_MANIFEST.json'
STATS_OUT = '/workspace/fwv1/p_items/FINGERTIP_TRAIN_NORM_STATS.json'
OUT = '/workspace/fwv1/p_items/FINGERTIP_FORCE_MUJOCO_ACCEPTANCE.json'
HIST_LEN = 8


def train_stats():
    split = json.load(open(SPLIT))
    train_ids = set(int(x) for x in split['train_episodes'])
    t = pq.read_table(PARQ, columns=['episode_index', 'observation.force'])
    E = np.asarray(t['episode_index'])
    fc = t['observation.force'].combine_chunks()
    F = fc.flatten().to_numpy(zero_copy_only=False).astype(np.float64).reshape(-1, 18)
    mask = np.isin(E, list(train_ids))
    Ftr = F[mask]
    pads = Ftr[:, list(CHANNELS)]
    stats = {'source': 'parquet force channels 12-15 over train split only',
             'split_manifest': SPLIT, 'n_train_episodes': len(train_ids),
             'n_train_frames': int(pads.shape[0]),
             'mean': [float(x) for x in pads.mean(0)],
             'std': [float(x) for x in pads.std(0)],
             'max_abs': [float(x) for x in np.abs(pads).max(0)],
             'frac_contact': [float(x) for x in (np.abs(pads) > 0.05).mean(0)],
             'inactive_channels': [i for i in range(4) if pads[:, i].std() <= 1e-6]}
    json.dump(stats, open(STATS_OUT, 'w'), indent=1)
    return stats


def make_processor(stats):
    return FingertipForceProcessor(stats['mean'], stats['std'], hist_len=HIST_LEN)


def synth_stream(n=60, seed=0):
    rng = np.random.default_rng(seed)
    F = rng.normal(0, 0.05, size=(n, 18))
    for i in range(n):
        F[i, list(CHANNELS)] = [0.0, 0.0, 0.0, 0.0] if i < 10 else rng.uniform(0, 1.5, 4)
    return F


def check_causality(stats):
    F = synth_stream()
    proc_full = make_processor(stats)
    full = []
    for i in range(len(F)):
        full.append(proc_full.update(F[i])['hist'].copy())
    proc_pref = make_processor(stats)
    pref = []
    for i in range(30):
        pref.append(proc_pref.update(F[i])['hist'].copy())
    max_diff = max(float(np.abs(full[i] - pref[i]).max()) for i in range(30))
    return {'hist_len': HIST_LEN, 'prefix_len': 30, 'max_prefix_diff': max_diff,
            'pass': bool(max_diff < 1e-6)}


def check_channel_isolation(stats):
    F1 = synth_stream(seed=1)
    F2 = F1.copy()
    rng = np.random.default_rng(2)
    F2[:, 0:12] = rng.normal(0, 50.0, size=(len(F2), 12))
    F2[:, 16:18] = rng.normal(0, 50.0, size=(len(F2), 2))
    p1 = make_processor(stats)
    p2 = make_processor(stats)
    max_diff = 0.0
    for i in range(len(F1)):
        h1 = p1.update(F1[i])['hist']
        h2 = p2.update(F2[i])['hist']
        max_diff = max(max_diff, float(np.abs(h1 - h2).max()))
    return {'perturbed_channels': '0-11 and 16-17', 'max_output_diff': max_diff,
            'pass': bool(max_diff < 1e-6)}


def check_invalid(stats):
    proc = make_processor(stats)
    F = synth_stream(n=20, seed=3)
    F[10, 12] = np.nan
    F[11, 13] = 1e9
    flags = []
    finite = True
    for i in range(len(F)):
        out = proc.update(F[i])
        flags.append((bool(out['valid']), bool(out['abnormal'])))
        finite = finite and bool(np.all(np.isfinite(out['hist'])))
    return {'frame10_nan_valid': flags[10][0], 'frame10_abnormal': flags[10][1],
            'frame11_huge_valid': flags[11][0], 'frame11_abnormal': flags[11][1],
            'frame9_valid': flags[9][0], 'hist_finite': finite,
            'pass': bool((not flags[10][0]) and flags[10][1] and (not flags[11][0])
                         and flags[11][1] and flags[9][0] and finite)}


def check_live(stats):
    from a3_dual_arm_sim import A3CookieTransferEnv, A3CookieTransferExpert
    env = A3CookieTransferEnv()
    u = getattr(env, 'unwrapped', env)
    obs, _ = env.reset()
    ex = A3CookieTransferExpert(env=u, action_mode='joint_position')
    ex.reset(context=None)
    proc = make_processor(stats)
    raw = []
    contact_pred = []
    contact_true = []
    valid = []
    hist_shapes = set()
    norm_absmax = 0.0
    sat_frames = 0
    first_contact = None
    info = {}
    max_force = np.zeros(4)
    for t in range(400):
        a = ex.act(obs, '')
        obs, r, term, trunc, info = env.step(np.asarray(a, dtype=float))
        f = np.asarray(obs['observation.force'], dtype=np.float64)
        out = proc.update(f)
        pads = f[list(CHANNELS)]
        raw.append(pads)
        contact_pred.append(out['contact_any'])
        contact_true.append(bool((np.abs(pads) > 0.05).any()))
        valid.append(out['valid'])
        hist_shapes.add(tuple(out['hist'].shape))
        norm_absmax = max(norm_absmax, float(np.abs(out['hist']).max()))
        if np.any(np.abs(out['hist']) >= 4.999):
            sat_frames += 1
        max_force = np.maximum(max_force, np.abs(pads))
        if first_contact is None and contact_true[-1]:
            first_contact = t
        if term or trunc:
            break
    if hasattr(env, 'close'):
        env.close()
    raw = np.asarray(raw)
    acc = float((np.asarray(contact_pred) == np.asarray(contact_true)).mean())
    dataset_max = float(max(stats['max_abs']))
    sim_max = float(max_force.max())
    return {
        'steps': len(raw),
        'first_contact_step': first_contact,
        'hist_shape_single': list(list(hist_shapes)[0]) if len(hist_shapes) == 1 else [list(x) for x in hist_shapes],
        'all_finite': bool(np.all(np.isfinite(raw))),
        'valid_rate': float(np.mean(valid)),
        'contact_agreement': acc,
        'pad_max_live': [float(x) for x in max_force],
        'normalized_absmax': norm_absmax,
        'saturation_frames': int(sat_frames),
        'saturation_frac': float(sat_frames / max(1, len(raw))),
        'dataset_pad_max': dataset_max,
        'live_to_dataset_scale': float(sim_max / dataset_max) if dataset_max > 0 else None,
        'cookies_in_target': info.get('cookies_in_target') if isinstance(info, dict) else None,
        'pass': bool(len(hist_shapes) == 1 and hist_shapes == {(HIST_LEN, 4)}
                     and np.mean(valid) > 0.99 and acc > 0.95),
    }


def main():
    res = {'worker': 'task7', 'ts': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
           'channels': list(CHANNELS), 'channel_names': ['L_inner', 'L_outer', 'R_inner', 'R_outer'],
           'hist_len': HIST_LEN, 'forbidden_channels': [16, 17]}
    try:
        stats = train_stats()
        res['train_stats'] = stats
        res['processor'] = 'p_items/scripts/t7_fingertip_processor.py (causal, 8-step, mask, clip=5)'
        res['tests'] = {
            'causality': check_causality(stats),
            'channel_isolation': check_channel_isolation(stats),
            'invalid_frames': check_invalid(stats),
            'live_mujoco': check_live(stats),
        }
        res['checks'] = {
            'train_stats_use_train_split_only': True,
            'four_fingertip_channels_only': True,
            'causality_prefix_equal': res['tests']['causality']['pass'],
            'channel_isolation': res['tests']['channel_isolation']['pass'],
            'invalid_frames_handled': res['tests']['invalid_frames']['pass'],
            'live_contact_agreement_gt_0.95': res['tests']['live_mujoco']['contact_agreement'] > 0.95,
            'hist_shape_8x4': res['tests']['live_mujoco']['hist_shape_single'] == [HIST_LEN, 4],
        }
        res['verdict'] = 'pass' if all(res['checks'].values()) else 'fail'
        live = res['tests']['live_mujoco']
        res['domain_gap_note'] = ('Dataset pad forces max %.2f N (train split), while the live MuJoCo expert run '
                                  'reaches %.2f N (%.1fx); normalization from dataset stats saturates %.3f of live frames. '
                                  'Training uses the dataset distribution; evaluation-time force inputs will be '
                                  'out-of-distribution and must be reported, not silently clipped.'
                                  % (float(max(stats['max_abs'])), float(max(live['pad_max_live'])),
                                     float(live['live_to_dataset_scale'] or 0.0),
                                     float(live['saturation_frac'])))
    except Exception:
        res['error'] = traceback.format_exc()[:2000]
        res['verdict'] = 'error'
    json.dump(res, open(OUT, 'w'), indent=1)
    print(json.dumps({'verdict': res['verdict'], 'checks': res.get('checks'),
                      'live': {k: res.get('tests', {}).get('live_mujoco', {}).get(k)
                               for k in ['steps', 'first_contact_step', 'contact_agreement',
                                         'pad_max_live', 'saturation_frac']}}))


if __name__ == '__main__':
    main()