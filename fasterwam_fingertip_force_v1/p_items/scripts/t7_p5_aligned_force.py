"""Aligned P5 replay force audit.

The previous P5 replay reset the env with seed=10000+episode and
randomize_cookies/randomize_boxes=False, which does NOT reproduce the scene
recorded in the episode metadata. This script resets with the recorded seed
and recorded reset_options, checks that the scene matches the recorded
cookie/bin positions, then replays the recorded actions and compares pad
forces against the dataset episode.
"""
import json
import os
import sys
import time
import traceback

os.environ.setdefault('MUJOCO_GL', 'egl')
sys.path.insert(0, '/workspace/fwv1/sim_repo/src')

import numpy as np
import pyarrow.parquet as pq

P = '/workspace/fwv1/p_items'
META = P + '/hf_multi/a3_episode_metadata.jsonl'
PARQ = [P + '/hf_multi/data/chunk-000/file-000.parquet',
        P + '/hf_multi/data/chunk-000/file-001.parquet']
OUT = P + '/P5_ALIGNED_FORCE_AUDIT.json'
EPISODES = [0, 422, 844, 17, 300, 700]
MAX_STEPS = 600


def q(a):
    a = np.asarray(a, np.float64).ravel()
    if a.size == 0:
        return None
    return {'n': int(a.size), 'mean': float(a.mean()), 'q50': float(np.quantile(a, 0.5)),
            'q99': float(np.quantile(a, 0.99)), 'max': float(a.max()),
            'contact_frac': float((np.abs(a) > 0.05).mean())}


def main():
    t0 = time.time()
    meta = {}
    for line in open(META):
        if line.strip():
            d = json.loads(line)
            meta.setdefault(int(d['episode_index']), []).append(d)
    t = pq.read_table(PARQ, columns=['episode_index', 'action', 'observation.state',
                                     'observation.force'])
    E = np.asarray(t['episode_index'])
    A = np.asarray(t['action'].to_pylist(), np.float64)
    S = np.asarray(t['observation.state'].to_pylist(), np.float64)
    F = t['observation.force'].combine_chunks().flatten().to_numpy(
        zero_copy_only=False).astype(np.float64).reshape(-1, 18)
    order = np.argsort(E, kind='stable')
    E, A, S, F = E[order], A[order], S[order], F[order]
    b = np.flatnonzero(np.diff(E)) + 1
    st = np.r_[0, b]
    en = np.r_[b, len(E)]
    starts = {int(E[st[k]]): int(st[k]) for k in range(len(st))}
    ends = {int(E[st[k]]): int(en[k]) for k in range(len(st))}

    from a3_dual_arm_sim import A3CookieTransferEnv

    records = []
    for ep in EPISODES:
        rec = {'episode': ep}
        try:
            s, e = starts[ep], ends[ep]
            n = e - s
            ents = meta.get(ep) or []
            cand = [x for x in ents if int(x.get('frames', -1)) == n and '2x5' in str(x.get('source_dataset'))]
            if not cand:
                cand = [x for x in ents if int(x.get('frames', -1)) == n]
            if not cand and ents:
                cand = [ents[0]]
            if not cand:
                raise RuntimeError('no metadata entry for episode')
            md = cand[0]
            rec['metadata_entries'] = len(ents)
            rec['source_dataset'] = md.get('source_dataset')
            rec['metadata_frames'] = md.get('frames')
            rnd = md.get('randomization') or {}
            ro = rnd.get('reset_options') or {}
            seed = int(md['seed'])
            rec['seed'] = seed
            rec['stored_action_mode'] = md.get('stored_action_mode')
            rec['source_action_mode'] = md.get('source_action_mode')
            rec['grasp_reports'] = len(rnd.get('grasp_reports') or [])
            rec['placement_order'] = rnd.get('placement_order')
            rec['target_column_order'] = rnd.get('target_column_order')
            env = A3CookieTransferEnv(action_mode='joint_position', render_mode='rgb_array',
                                      render_cameras=False)
            u = getattr(env, 'unwrapped', env)
            obs, info = env.reset(seed=seed, options=ro)
            rm = dict(getattr(u, 'randomization_metadata', {}) or {})

            def md_diff(key):
                a = np.asarray(rnd.get(key), np.float64)
                b2 = np.asarray(rm.get(key), np.float64)
                if a.size == 0 or a.size != b2.size:
                    return None
                return float(np.max(np.abs(a - b2)))

            rec['scene_diff'] = {'cookie_positions': md_diff('cookie_positions'),
                                 'source_bin_position': md_diff('source_bin_position'),
                                 'target_bin_position': md_diff('target_bin_position'),
                                 'target_bin_quaternion': md_diff('target_bin_quaternion')}
            s, e = starts[ep], ends[ep]
            rec['dataset_frames'] = int(e - s)
            obs0 = np.asarray(obs['observation.state'], np.float64)
            if obs0.size == 16 and S[s].size == 8:
                obs0 = obs0[:8]
            rec['initial_state_max_abs_diff'] = float(np.max(np.abs(obs0 - S[s]))) if obs0.size == S[s].size else None
            home = np.asarray(u.DEPLOYMENT_HOME, np.float64)
            right_arm = home[8:15].copy()
            right_grip = float(home[15])
            n = min(e - s, MAX_STEPS)
            forces = []
            states = []
            for k in range(n):
                a16 = np.r_[A[s + k, :7], A[s + k, 7], right_arm, right_grip]
                obs, r, term, trunc, info = env.step(a16)
                forces.append(np.asarray(obs['observation.force'], np.float64)[12:16])
                stv = np.asarray(obs['observation.state'], np.float64)
                states.append(stv[:8] if stv.size == 16 else stv)
                if term or trunc:
                    break
            forces = np.asarray(forces, np.float64)
            rec['replay_steps'] = int(len(forces))
            rec['force_dataset'] = {'ch12': q(F[s:s + len(forces), 12]), 'ch13': q(F[s:s + len(forces), 13]),
                                    'ch14': q(F[s:s + len(forces), 14]), 'ch15': q(F[s:s + len(forces), 15])}
            rec['force_replay'] = {'ch12': q(forces[:, 0]), 'ch13': q(forces[:, 1]),
                                   'ch14': q(forces[:, 2]), 'ch15': q(forces[:, 3])}
            rec['force_ratio_q99'] = {
                'ch12': (rec['force_replay']['ch12']['q99'] / rec['force_dataset']['ch12']['q99'])
                if rec['force_dataset']['ch12']['q99'] else None,
                'ch13': (rec['force_replay']['ch13']['q99'] / rec['force_dataset']['ch13']['q99'])
                if rec['force_dataset']['ch13']['q99'] else None}
            rec['state_track_max_abs_diff'] = float(np.max(np.abs(np.asarray(states) - S[s:s + len(states)]))) if states else None
            rec['terminated'] = bool(term)
            rec['truncated'] = bool(trunc)
            env.close()
        except Exception:
            rec['error'] = traceback.format_exc()[:1500]
        records.append(rec)
        print(json.dumps({k: rec.get(k) for k in ('episode', 'seed', 'scene_diff',
                                                  'initial_state_max_abs_diff',
                                                  'force_dataset', 'force_replay',
                                                  'force_ratio_q99', 'error')}), flush=True)
    ok = [x for x in records if not x.get('error')]
    ratios = [x['force_ratio_q99']['ch12'] for x in ok if x.get('force_ratio_q99', {}).get('ch12')]
    ratios += [x['force_ratio_q99']['ch13'] for x in ok if x.get('force_ratio_q99', {}).get('ch13')]
    scene_ok = all((x.get('scene_diff') or {}).get('cookie_positions') is not None and
                   (x.get('scene_diff') or {}).get('cookie_positions') < 1e-6 for x in ok)
    result = {'worker': 'supervisor', 'ts': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
              'purpose': 'verify whether the 15-19x dataset-vs-replay force gap was caused by an unaligned scene',
              'protocol': {'reset': 'seed=metadata.seed, options=metadata.randomization.reset_options',
                           'action_map': 'dataset 8D [L 7, L grip] -> env 16D [L 7, L grip, R home 7, R grip home]',
                           'max_steps': MAX_STEPS},
              'episodes': records,
              'summary': {'scene_reproduced': scene_ok,
                          'force_ratio_q99_median': float(np.median(ratios)) if ratios else None,
                          'force_ratio_q99_max': float(np.max(ratios)) if ratios else None,
                          'verdict': 'aligned_force_matches' if (scene_ok and ratios and np.median(ratios) < 2.0)
                          else ('scene_mismatch_or_force_gap_persists')},
              'elapsed_s': round(time.time() - t0, 1)}
    json.dump(result, open(OUT, 'w'), indent=1)
    print('DONE ' + result['summary']['verdict'], flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        print('TOPFAIL', flush=True)
        print(traceback.format_exc()[:2500], flush=True)