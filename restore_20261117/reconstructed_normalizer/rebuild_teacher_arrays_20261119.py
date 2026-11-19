"""Rebuild the teacher-replay arrays with correct episode offsets and deploy-identical proprio.

The recorded frames.dat/actions/forces are intact; only arrays.npz's ep_start accounting
(`len(frames)` = episode count instead of cumulative frames) and its proprio construction were
wrong (75% zero rows, deployment uses base8+force6). This script re-runs the same four expert
episodes on CPU without cameras to regenerate poses/opening/proprio, and gates on the
regenerated force18 and action streams matching the stored ones frame-for-frame.

Output: teacher_replay_20261119/ with a symlink to the original frames.dat.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

SIM_SRC = '/workspace/eval_restore_20261117/deploy/a3_dual_arm_sim/src'
if SIM_SRC not in sys.path:
    sys.path.insert(0, SIM_SRC)

import diag_reconstructed_40k as D

ROOT = Path(__file__).resolve().parent
SRC = ROOT / 'teacher_replay_20261118'
OUT = ROOT / 'teacher_replay_20261119'
SEEDS = [41, 42, 43, 44]
MAX_STEPS = 600

E, factory, W, fw = D.setup_protocol()
from a3_dual_arm_sim.controllers.batch_expert import A3CookieBatchExpert
calib = E.read_json(str(ROOT.parent / 'step_040000' / 'force_calib_v1.json'))


def rotvec_between(Ra, Rb):
    R = Rb @ Ra.T
    c = np.clip((np.trace(R) - 1) / 2, -1, 1)
    ang = np.arccos(c)
    if ang < 1e-8:
        return np.zeros(3)
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return axis / (2 * np.sin(ang)) * ang


def quat_wxyz_to_mat(q):
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def grip_labels(opening, eps=0.02, min_hold=3):
    L = len(opening)
    d = np.diff(opening)
    lab = np.empty(L)
    prev = 1.0
    for t in range(L - 1):
        if d[t] < -eps:
            prev = 1.0
        elif d[t] > eps:
            prev = -1.0
        lab[t] = prev
    lab[L - 1] = lab[L - 2] if L > 1 else 1.0
    for _ in range(3):
        i = 0
        while i < L:
            j = i
            while j < L and lab[j] == lab[i]:
                j += 1
            if j - i < min_hold:
                lab[i:j] = lab[i - 1] if i > 0 else lab[i]
            i = j
    return lab


def main():
    old = np.load(SRC / 'arrays.npz', allow_pickle=True)
    of = np.load(SRC / 'force_arrays.npz', allow_pickle=True)
    old_act = old['action'].astype(np.float64)
    old_f18 = of['force18'].astype(np.float64)
    lens_old = [int(x) for x in old['ep_len']]

    OUT.mkdir(exist_ok=True)
    frames_link = OUT / 'frames.dat'
    if not frames_link.exists():
        os.symlink(str(SRC / 'frames.dat'), frames_link)

    actions, propios, forces, poses_all, opens_all = [], [], [], [], []
    starts, lens, gates = [], [], []
    offset = 0
    for k, seed in enumerate(SEEDS):
        env = factory.make_env(position_noise_m=0.0, yaw_noise_rad=0.0, render_cameras=False)
        reset = D.make_reset(E, factory, env, type('A', (), {
            'cpu': True, 'gripper_mode': 'pm1', 'max_steps': MAX_STEPS, 'replan': 10,
            'allow_invalid_scene': False})())
        obs = reset(env, seed)
        u = getattr(env, 'unwrapped', env)
        expert = A3CookieBatchExpert(env=u, action_mode='joint_position')
        expert.reset(context=None)
        poses = [np.asarray(obs['observation.eef_pose'], np.float64)]
        ops = [float(np.asarray(obs['observation.state'])[7])]
        props = [E.build_proprio14(obs, W, calib, where='rebuild')]
        f_step = [np.asarray(obs['observation.force'], np.float64)]
        for t in range(MAX_STEPS):
            a = expert.act(obs, '')
            out = env.step(np.asarray(a, np.float64))
            obs = out[0]
            poses.append(np.asarray(obs['observation.eef_pose'], np.float64))
            ops.append(float(np.asarray(obs['observation.state'])[7]))
            props.append(E.build_proprio14(obs, W, calib, where='rebuild'))
            f_step.append(np.asarray(obs['observation.force'], np.float64))
        if hasattr(env, 'close'):
            env.close()
        poses = np.asarray(poses)
        ops = np.asarray(ops)
        act = np.zeros((len(poses) - 1, 7))
        for t in range(len(poses) - 1):
            act[t, :3] = (poses[t + 1][:3] - poses[t][:3]) / 0.05
            act[t, 3:6] = rotvec_between(quat_wxyz_to_mat(poses[t][3:7]),
                                         quat_wxyz_to_mat(poses[t + 1][3:7])) / 0.5
        act[:, :6] = np.clip(act[:, :6], -1, 1)
        act[:, 6] = grip_labels(ops)[:len(act)]
        f18 = np.asarray(f_step[:-1])
        n = len(act)
        old_a = old_act[offset:offset + n]
        old_f = old_f18[offset:offset + n]
        gate = {
            'seed': seed, 'frames': n, 'old_len': lens_old[k],
            'action_max_abs_diff': float(np.abs(act - old_a).max()),
            'force_max_abs_diff': float(np.abs(f18 - old_f).max()),
            'force_bit_equal': bool(np.array_equal(f18, old_f)),
        }
        gate['deterministic_replay_ok'] = bool(gate['force_bit_equal']
                                               and gate['action_max_abs_diff'] < 1e-9)
        gates.append(gate)
        print(gate, flush=True)
        starts.append(offset)
        lens.append(n)
        actions.append(act.astype(np.float32))
        propios.append(np.asarray(props[:-1], np.float32))
        forces.append(f18.astype(np.float32))
        poses_all.append(poses)
        opens_all.append(ops[:-1].astype(np.float32))
        offset += n

    if not all(g['deterministic_replay_ok'] for g in gates):
        (OUT / 'REBUILD_GATE_FAILED.json').write_text(json.dumps(gates, indent=2))
        raise SystemExit('DETERMINISM_GATE_FAILED; refusing to write arrays')

    act_all = np.concatenate(actions)
    prop_all = np.concatenate(propios)
    f18_all = np.concatenate(forces)
    np.savez_compressed(OUT / 'arrays.npz', action=act_all, proprio=prop_all,
                        ep_start=np.asarray(starts, np.int64),
                        ep_len=np.asarray(lens, np.int64),
                        EXPERT_ACTIONS=np.asarray(True), **D.MARKERS)
    cal = json.load(open(ROOT.parent / 'step_040000' / 'force_calib_v1.json'))
    fc = [int(c) for c in cal['channels']]
    np.savez_compressed(OUT / 'force_arrays.npz', force18=f18_all,
                        force6=f18_all[:, fc],
                        ep_start=np.asarray(starts, np.int64),
                        ep_len=np.asarray(lens, np.int64))
    np.savez_compressed(OUT / 'poses.npz',
                        **{'poses_%d' % s: p for s, p in zip(SEEDS, poses_all)},
                        **{'opening_%d' % s: o for s, o in zip(SEEDS, opens_all)})

    st = json.load(open(ROOT / 'dataset_stats_14d_RECONSTRUCTED.json'))['state']['default']
    pmean = np.asarray(st['global_mean'])
    pstd = np.asarray(st['global_std'])
    z = (prop_all.mean(0) - pmean) / np.where(pstd > 1e-9, pstd, np.nan)
    man = {
        **D.MARKERS, 'source_dir': str(SRC), 'seeds': SEEDS, 'frames': int(len(act_all)),
        'ep_start': starts, 'ep_len': lens, 'EXPERT_ACTIONS': True,
        'proprio_layout': 'deployment build_proprio14 = base8 + force-tare residuals',
        'windows_num_frames33': int(sum(L - 33 + 1 for L in lens)),
        'gates': gates,
        'proprio_mean': prop_all.mean(0).tolist(), 'proprio_std': prop_all.std(0).tolist(),
        'proprio_mean_abs_z_vs_recon': np.abs(z).tolist(),
        'max_abs_z': float(np.nanmax(np.abs(z))),
        'bug_fixed': 'ep_start cumulative frames; proprio columns 8..13 force residuals; '
                     'columns 6..7 real finger joints from the deployment builder',
    }
    (OUT / 'manifest.json').write_text(json.dumps(man, indent=2))
    print(json.dumps({k: man[k] for k in ('frames', 'ep_start', 'ep_len', 'max_abs_z',
                                          'windows_num_frames33')}, indent=2))
    print('REBUILD_DONE', flush=True)


if __name__ == '__main__':
    main()
