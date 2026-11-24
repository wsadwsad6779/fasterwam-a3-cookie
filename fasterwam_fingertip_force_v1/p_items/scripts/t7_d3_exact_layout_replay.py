"""D3 exact-layout replay: set the recorded scene explicitly, then replay the
recorded joint actions. This removes the RNG/scene-reproduction uncertainty and
decides whether dataset and current sim/sensor/control agree on force scale.
"""
import json
import os
import sys
import time

os.environ.setdefault('MUJOCO_GL', 'egl')
sys.path.insert(0, '/workspace/fwv1/sim_repo/src')

import mujoco
import numpy as np
import pyarrow.parquet as pq

P = '/workspace/fwv1/p_items'
META = P + '/hf_multi/a3_episode_metadata.jsonl'
PARQ = [P + '/hf_multi/data/chunk-000/file-000.parquet',
        P + '/hf_multi/data/chunk-000/file-001.parquet']
OUT = P + '/D3_EXACT_LAYOUT_REPLAY.json'
EPISODE = 0


def q(a):
    a = np.asarray(a, np.float64).ravel()
    if a.size == 0:
        return None
    return {'n': int(a.size), 'mean': float(a.mean()), 'q50': float(np.quantile(a, 0.5)),
            'q90': float(np.quantile(a, 0.9)), 'q99': float(np.quantile(a, 0.99)),
            'max': float(a.max()), 'contact_frac': float((np.abs(a) > 0.05).mean())}


def main():
    meta = {}
    for line in open(META):
        if line.strip():
            d = json.loads(line)
            meta.setdefault(int(d['episode_index']), []).append(d)
    t = pq.read_table(PARQ, columns=['episode_index', 'action', 'observation.force'])
    E = np.asarray(t['episode_index'])
    A = np.asarray(t['action'].to_pylist(), np.float64)
    F = t['observation.force'].combine_chunks().flatten().to_numpy(
        zero_copy_only=False).astype(np.float64).reshape(-1, 18)
    order = np.argsort(E, kind='stable')
    E, A, F = E[order], A[order], F[order]
    b = np.flatnonzero(np.diff(E)) + 1
    st = np.r_[0, b]
    en = np.r_[b, len(E)]
    starts = {int(E[st[k]]): int(st[k]) for k in range(len(st))}
    ends = {int(E[st[k]]): int(en[k]) for k in range(len(st))}
    s, e = starts[EPISODE], ends[EPISODE]
    n = e - s
    ents = meta[EPISODE]
    cand = [x for x in ents if int(x.get('frames', -1)) == n and '2x5' in str(x.get('source_dataset'))]
    md = cand[0]
    rnd = md['randomization']
    ro = rnd['reset_options']
    seed = int(md['seed'])
    cp = np.asarray(rnd['cookie_positions'], np.float64)
    cq = np.asarray(rnd['cookie_quaternions'], np.float64)
    sb = np.asarray(rnd['source_bin_position'], np.float64)
    tb = np.asarray(rnd['target_bin_position'], np.float64)
    tq = np.asarray(rnd['target_bin_quaternion'], np.float64)

    from a3_dual_arm_sim import A3CookieTransferEnv
    env = A3CookieTransferEnv(action_mode='joint_position', render_mode='rgb_array',
                              render_cameras=False)
    u = getattr(env, 'unwrapped', env)
    env.reset(seed=seed, options=ro)
    jid = mujoco.mj_name2id(u.model, mujoco.mjtObj.mjOBJ_JOINT, 'target_bin_free')
    qadr = int(u.model.jnt_qposadr[jid])
    dadr = int(u.model.jnt_dofadr[jid])
    u.data.qpos[qadr:qadr + 3] = tb
    u.data.qpos[qadr + 3:qadr + 7] = tq
    u.data.qvel[dadr:dadr + 6] = 0.0
    u.model.body_pos[u._source_bin_body, 0] = sb[0]
    u.model.body_pos[u._source_bin_body, 1] = sb[1]
    for i in range(cp.shape[0]):
        u.set_cookie_pose(i, cp[i], cq[i])
    mujoco.mj_forward(u.model, u.data)
    after_cp = np.asarray(u.cookie_positions, np.float64)
    scene = {
        'cookie_max_abs_diff': float(np.max(np.abs(after_cp - cp))) if after_cp.shape == cp.shape else None,
        'source_bin_diff': float(np.max(np.abs(u.data.xpos[u._source_bin_body][:2] - sb[:2]))),
        'target_bin_diff': float(np.max(np.abs(u.data.xpos[u._target_bin_body][:2] - tb[:2]))),
    }
    home = np.asarray(u.DEPLOYMENT_HOME, np.float64)
    ra = home[8:15].copy()
    rg = float(home[15])
    forces = []
    openings = []
    eefs = []
    info = {}
    for k in range(n):
        a16 = np.r_[A[s + k, :7], A[s + k, 7], ra, rg]
        obs, r, term, trunc, info = env.step(a16)
        forces.append(np.asarray(obs['observation.force'], np.float64)[12:16])
        openings.append(float(u._gripper_state('L')[0]))
        eefs.append(np.asarray(obs['observation.eef_pose'], np.float64).reshape(-1)[:3])
        if term or trunc:
            break
    forces = np.asarray(forces, np.float64)
    eefs = np.asarray(eefs, np.float64)
    df = F[s:s + len(forces), 12:16]
    path = float(np.sum(np.linalg.norm(np.diff(eefs, axis=0), axis=1))) if len(eefs) > 1 else 0.0
    result = {'worker': 'supervisor', 'ts': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
              'episode': EPISODE, 'seed': seed, 'steps': len(forces), 'scene': scene,
              'force_dataset': {'ch12': q(df[:, 0]), 'ch13': q(df[:, 1])},
              'force_replay': {'ch12': q(forces[:, 0]), 'ch13': q(forces[:, 1])},
              'ratio_q99': {'ch12': float(q(forces[:, 0])['q99'] / q(df[:, 0])['q99']) if q(df[:, 0])['q99'] else None,
                            'ch13': float(q(forces[:, 1])['q99'] / q(df[:, 1])['q99']) if q(df[:, 1])['q99'] else None},
              'actual_opening': {'min': float(np.min(openings)), 'max': float(np.max(openings)),
                                 'close_frac': float(np.mean(np.asarray(openings) < 0.5))},
              'eef_path_m': path,
              'final_in_target': info.get('cookies_in_target'),
              'final_in_source': info.get('cookies_in_source'),
              'safety_reason': info.get('safety_reason'),
              'verdict': 'dataset_and_current_sim_agree' if (scene['cookie_max_abs_diff'] is not None and
                                                              scene['cookie_max_abs_diff'] < 1e-5 and
                                                              all(v is not None and v < 2.0 for v in
                                                                  [q(forces[:, 0])['q99'] / q(df[:, 0])['q99'] if q(df[:, 0])['q99'] else None,
                                                                   q(forces[:, 1])['q99'] / q(df[:, 1])['q99'] if q(df[:, 1])['q99'] else None]))
              else 'investigate'}
    json.dump(result, open(OUT, 'w'), indent=1)
    print(json.dumps(result, indent=1)[:4000], flush=True)
    env.close()


if __name__ == '__main__':
    main()