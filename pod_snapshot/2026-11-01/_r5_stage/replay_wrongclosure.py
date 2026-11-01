'''C-2 correction demos: the scripted expert wrongly closes on a whole batch, then
re-plans onto a single cookie and finishes a validated low-force grasp.

The anomaly is the scenes own, measured in replay_bundle_dist: A3SameColumnBatchExpert
takes 5 cookies per clamp at closure peaks of only 1.32-1.54 N, lifts 81 mm and releases
upright -- it passes every force and placement criterion and is rejected solely because
max_cookies_at_grip != 1.  That is exactly the 'wrong closure' state the task wants a
correction demo for, and it is invisible to a force-only trigger.

The correction is an explicit release followed by the expert controllers own
re-initialisation (a fresh SingleCookieLowForce in the same MuJoCo scene): during the
OBSERVE_FRAMES hold window the gripper channel of the action is pinned to RELEASE_OPEN so
the wrongly held batch actually leaves the gripper.  The channel is dim 6 of the 14-dim
delta-TCP action the experts build (dim 7 belongs to the idle right arm, and the first
version of this driver pinned that one -- the fingers never moved and the batch left the
gripper only because the re-planned controllers own approach jerked it out).  Pinning is
not cosmetic either -- the version that left the release to chance worked for four seeds
but for seed 7037 kept all five cookies clamped while descending onto the recovery column
and pressed 9.9 N for 398 frames before timing out.  The opening command is rate-limited to
0.08 per step, so 0.31 -> 1.00 needs nine steps plus finger lag; the window is 16 frames.
Nothing here changes the model, the Wrapper, the Safety layer or the simulator; the
environment still executes whatever action the replay driver hands it.

Reuses /workspace/_r5_stage/replay_forces.py for the environment, the low-force policy,
the phase segmentation and the per-clamp acceptance statistics.'''

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np

import replay_forces as R

WRONG_COUNT = 2           # cookies in the gripper that make a closure 'wrong' for this task
ANOMALY_PAD_N = 1.6       # corpus absolute maximum over 42681 expert frames is 1.540 N
HARD_CAP_N = 2.0
OBSERVE_FRAMES = 16       # keep driving after the wrong closure is seen, so it is observable
RELEASE_OPEN = 1.0        # opening that empties the gripper, encoded below into GRIPPER_ACT_DIM
GRIPPER_ACT_DIM = 6       # 14-dim delta-TCP action: dP(0:3) dR(3:6) grip(6), right arm 7:14
RELEASE_ACT_DIM6 = 2.0 * RELEASE_OPEN - 1.0   # the experts own encoding in batch_expert._servo


def cookies_at_grip(xy, z, z0, eef_xy):
    rise = z - z0[:, 2]
    return int(np.sum((np.linalg.norm(xy - eef_xy, axis=1) < R.HOLD_XY_M) & (rise > R.LIFT_Z_M)))


def run_wrongclosure(seed, recover, max_steps, noise, out_dir, wrong_count=WRONG_COUNT,
                     hold_frames=OBSERVE_FRAMES):
    env, obs, info = R.make_env(noise, seed)
    expert = R.A3SameColumnBatchExpert(env)      # corpus default batch behaviour: multi-cookie
    expert.reset()
    z0 = np.asarray(env.cookie_positions, dtype=np.float64).copy()
    z_rest = float(np.max(env.cookie_positions[:, 2]))

    phase, pad, xyz, eef, opening, applied, f18, wrist, at_grip = [], [], [], [], [], [], [], [], []

    def tick(action):
        nonlocal obs, info
        obs, _, terminated, truncated, info = env.step(action)
        f = np.asarray(obs[R.FORCE_KEY], dtype=np.float64)
        cur_xyz = np.asarray(env.cookie_positions, dtype=np.float64)
        cur_eef = np.asarray(obs['observation.eef_pose'][:3], dtype=np.float64)
        phase.append(expert.phase.name)
        pad.append(float(max(f[R.L_INNER], f[R.L_OUTER])))
        xyz.append(cur_xyz.copy())
        eef.append(cur_eef.copy())
        opening.append(float(obs['observation.state'][7]))
        applied.append(float(np.asarray(info['applied_action'])[7]))
        f18.append(f.copy())
        wrist.append(f[0:3].copy())
        at_grip.append(cookies_at_grip(cur_xyz[:, :2], cur_xyz[:, 2], z0, cur_eef[:2]))
        return terminated, truncated

    retry_log = []
    variant = None
    t0, stop = time.time(), None
    for step in range(max_steps):
        terminated, truncated = tick(expert.act(obs))
        if terminated:
            stop = 'env_safety_stop: ' + str(info.get('safety_reason'))
            break
        if truncated:
            stop = 'horizon'
            break
        if expert.failed:
            retry_log.append({'at_step': len(pad), 'attempt': 'bundle',
                              'reason': str(expert.failure_reason)[:120],
                              'anomaly_kind': 'bundle_transport_failure',
                              'peak_pad_n': round(float(max(pad)), 3),
                              'max_at_grip': int(max(at_grip))})
            stop = 'bundle_failed_before_correction'
            break

        # Wrong-closure trigger: the batch is lifted, held, and it is more than one cookie.
        if (not retry_log and expert.phase.name in R.CARRY_PHASES
                and at_grip[-1] >= wrong_count):
            seen = len(pad)
            while len(pad) < seen + hold_frames:
                act = expert.act(obs)
                act[GRIPPER_ACT_DIM] = RELEASE_ACT_DIM6
                terminated, truncated = tick(act)
                if terminated or truncated or expert.failed or expert.finished:
                    break
            retry_log.append({
                'at_step': len(pad), 'attempt': 'bundle',
                'reason': 'batch closure holds %d cookies, task wants 1' % int(max(at_grip)),
                'anomaly_kind': 'wrong_closure',
                'max_at_grip': int(max(at_grip)),
                'peak_pad_n': round(float(max(pad)), 3),
                'closure_forces_are_low': bool(max(pad) <= ANOMALY_PAD_N),
                'extras_released_in_correction': bool(min(at_grip[seen:]) <= 1),
            })
            variant = int(recover[0])
            expert = R.SingleCookieLowForce(env)
            expert.variant_index = variant
            expert.reset()
            continue

        if expert.finished:
            stop = 'expert_done'
            break

    xyz = np.asarray(xyz)
    eef = np.asarray(eef)
    f18 = np.asarray(f18)
    pad = np.asarray(pad)
    opening = np.asarray(opening)
    at_grip = np.asarray(at_grip)
    np.savez_compressed(
        out_dir / ('expert_wrongclosure_%d.npz' % int(seed)),
        cookie_xyz=xyz, cookie_xyz0=z0, eef_xyz=eef, opening=opening,
        applied_opening=np.asarray(applied), pad_force_inner=f18[:, R.L_INNER],
        pad_force_outer=f18[:, R.L_OUTER], wrist_force=np.asarray(wrist), force18=f18,
        cookies_at_grip=at_grip, changed=np.zeros(len(xyz), dtype=bool),
        phase=np.asarray([R.PHASE_NAMES.index(p) for p in phase], dtype=np.int8),
        attempts=np.asarray([int(variant)] if variant is not None else [], dtype=np.int64),
    )
    clamps = [R.clamp_stats(a, b, phase, pad, xyz, eef, z0, opening, z_rest)
              for a, b in R.segments(phase)]
    cut = int(retry_log[0]['at_step']) if retry_log else len(pad)
    rec = pad[cut:]
    segs = {
        'wrong_half': {'n': int(cut), 'max_at_grip': int(at_grip[:cut].max()) if cut else 0,
                       'pad_peak': round(float(pad[:cut].max()), 3) if cut else 0.0},
        'recovery_half': {'n': int(len(rec)),
                          'pad_peak': round(float(rec.max()), 3) if len(rec) else 0.0,
                          'frames_over_1p6': int(np.sum(rec > ANOMALY_PAD_N)),
                          'frames_over_2n': int(np.sum(rec > HARD_CAP_N))},
    }
    rest = xyz[-1]
    inbin = ((np.abs(rest[:, 0] - R.BIN_XY[0]) < R.BIN_INNER_HALF[0])
             & (np.abs(rest[:, 1] - R.BIN_XY[1]) < R.BIN_INNER_HALF[1]))
    moved = np.linalg.norm(rest - z0, axis=1) > 0.05
    upright = np.abs(rest[:, 2] - z0[:, 2]) < 0.004
    last = clamps[-1] if clamps else {}
    wrong_observed = bool(retry_log and retry_log[0]['anomaly_kind'] == 'wrong_closure'
                          and retry_log[0]['max_at_grip'] >= wrong_count)
    recovery_clean = (segs['recovery_half']['frames_over_2n'] == 0
                      and segs['recovery_half']['pad_peak'] <= ANOMALY_PAD_N)
    grasp_valid = bool(last.get('accept')) and int(last.get('max_cookies_at_grip', 0)) == 1
    return {
        'seed': int(seed), 'mode': 'wrongclosure', 'noise_m': noise,
        'recover_variant': variant, 'steps': len(phase), 'stop': stop,
        'success': bool(info.get('success', False)),
        'in_target_final': int(np.sum(inbin & moved)),
        'moved_into_bin_upright': int(np.sum(inbin & moved & upright)),
        'pad_p50_n': round(float(np.percentile(pad, 50)), 3),
        'pad_max_n': round(float(pad.max()), 3),
        'frames_over_2n': int(np.sum(pad > HARD_CAP_N)),
        'max_cookies_at_grip': int(at_grip.max()),
        'retry_log': retry_log,
        'segments': segs,
        'wrong_closure_observed': bool(wrong_observed),
        'recovery_clean': bool(recovery_clean),
        'grasp_valid': bool(grasp_valid),
        'accept': bool(wrong_observed and recovery_clean and grasp_valid),
        'clamps': clamps,
        'wall_s': round(time.time() - t0, 1),
    }


def main():
    ap = argparse.ArgumentParser(description='C-2 wrong-closure demos')
    ap.add_argument('--seeds', type=int, nargs='+', required=True)
    ap.add_argument('--recover-variants', type=int, nargs='+', default=[0])
    ap.add_argument('--max-steps', type=int, default=1600)
    ap.add_argument('--noise', type=float, default=0.0003)
    ap.add_argument('--wrong-count', type=int, default=WRONG_COUNT)
    ap.add_argument('--hold-frames', type=int, default=OBSERVE_FRAMES)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists() and any(out.glob('*')):
        raise SystemExit('refusing to write into a non-empty directory: ' + str(out))
    free = shutil.disk_usage(str(out.parent if out.parent.exists() else '/workspace')).free
    if free < 1_000_000_000:
        raise SystemExit('only %d bytes free, refusing to start' % free)
    out.mkdir(parents=True, exist_ok=True)
    print('out=%s free_bytes=%d driver_sha=%s policy_sha=%s' % (
        out, free,
        hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        hashlib.sha256(Path('/workspace/_r5_stage/replay_forces.py').read_bytes()).hexdigest()),
        flush=True)

    rows = []
    for seed in args.seeds:
        row = run_wrongclosure(seed, args.recover_variants, args.max_steps, args.noise, out,
                               args.wrong_count, args.hold_frames)
        rows.append(row)
        print(json.dumps({k: row[k] for k in ('seed', 'steps', 'stop', 'recover_variant',
                                              'max_cookies_at_grip', 'pad_max_n',
                                              'frames_over_2n', 'wrong_closure_observed',
                                              'recovery_clean', 'grasp_valid', 'accept',
                                              'in_target_final')}, default=float), flush=True)
        for r in row['retry_log']:
            print('      pre', json.dumps(r), flush=True)
        print('      segs', json.dumps(row['segments'], default=float), flush=True)
        for c in row['clamps']:
            print('      clamp', json.dumps({k: c.get(k) for k in
                                             ('phases', 'closure_peak_pad_n',
                                              'carry_peak_pad_n', 'post_release_pad_n',
                                              'max_cookies_at_grip', 'carried_lift_mm',
                                              'placed_in_bin_xy', 'placed_upright',
                                              'accept')}, default=float), flush=True)
        (out / 'summary.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')


main()
