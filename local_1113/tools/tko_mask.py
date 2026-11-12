"""Replay the recorded executed actions through the simulator, read-only.

Purpose: recover the per-step `cookies_in_target_mask` and attribute every
Safety rewrite to its rule, without touching run_matrix or the frozen outputs.
The replay is only trusted after its cookie trajectory and its Safety output
match the npz.

    python3 tko_mask.py --npz <run.npz> --seed 22 --out <mask.npz> --movers 0 1 2 3 4 5
"""
import argparse
import json
from pathlib import Path

import numpy as np

import env_factory
import safety_wrap


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--movers", type=int, nargs="*", default=[])
    return ap.parse_args()


def main():
    args = parse_args()
    m = np.load(args.npz, allow_pickle=False)
    pre = m["pre_14d"].astype(np.float64)
    post_stored = m["post_14d"].astype(np.float64)
    xyz_stored = m["cookie_xyz"].astype(np.float64)
    changed_stored = m["changed"].astype(bool)
    T = len(pre)
    N = xyz_stored.shape[1]

    env = env_factory.make_env(position_noise_m=0.0, yaw_noise_rad=0.0,
                               render_cameras=False)
    obs, info = env_factory.reset(env, args.seed, randomize_cookies=True,
                                  randomize_boxes=False, batch_offset_m=0.0)

    geom = {
        "target_bin_xpos": np.round(env.data.xpos[env._target_bin_body], 5).tolist(),
        "TARGET_INNER_HALF_SIZE": np.round(np.asarray(env.TARGET_INNER_HALF_SIZE), 5).tolist(),
        "TARGET_WALL_HEIGHT": float(env.TARGET_WALL_HEIGHT),
        "TARGET_FLOOR_TOP_Z": float(env.TARGET_FLOOR_TOP_Z),
        "COOKIE_HALF_SIZE": np.round(np.asarray(env.COOKIE_HALF_SIZE), 5).tolist(),
        "cookie_z0": sorted(set(np.round(np.asarray(env.cookie_positions)[:, 2], 5).tolist())),
        "required_cookies": int(env.task_config.required_cookies),
        "require_released": bool(env.task_config.require_released),
        "max_linear_speed": float(env.task_config.max_linear_speed_m_s),
        "max_tilt_rad": float(env.task_config.max_tilt_rad),
    }
    print("GEOM " + json.dumps(geom), flush=True)

    lim = safety_wrap.SafetyLimits()
    st = safety_wrap.new_state()
    mask = np.zeros((T, N), dtype=bool)
    traj = np.zeros((T, N, 3))
    count = np.zeros(T, dtype=np.int64)
    rule = {k: np.zeros(T, dtype=np.int64) for k in ("hard", "soft", "binfloor", "clipped")}
    contact_any = np.zeros((T, N), dtype=np.int8)
    post_replayed = np.zeros_like(pre)
    eef = np.zeros((T, 3))
    open_arr = np.zeros(T)
    app_open_arr = np.zeros(T)
    force_arr = np.zeros((T, 2))

    last = T - 1
    for t in range(T):
        before = {k: st[k] for k in rule}
        a3 = safety_wrap.apply(pre[t], obs, st, lim)
        post_replayed[t] = a3
        for k in rule:
            rule[k][t] = st[k] - before[k]

        obs, _, terminated, truncated, info = env.step(a3)
        eef[t] = np.asarray(obs["observation.eef_pose"][:3], dtype=np.float64)
        open_arr[t] = float(obs["observation.state"][7])
        app_open_arr[t] = float(info["applied_action"][7])
        f2 = np.asarray(obs[safety_wrap.FORCE_KEY], dtype=np.float64)
        force_arr[t] = (f2[safety_wrap.L_INNER], f2[safety_wrap.L_OUTER])
        traj[t] = np.asarray(env.cookie_positions, dtype=np.float64)
        mask[t] = np.asarray(info["cookies_in_target_mask"], dtype=bool)
        count[t] = int(info["cookies_in_target"])
        for i in args.movers:
            ci, co = env.privileged_left_finger_contacts(int(i))
            contact_any[t, i] = int(ci or co)
        last = t
        if terminated or truncated:
            print(f"episode ended early at t={t} terminated={terminated} truncated={truncated}",
                  flush=True)
            break

    sl = slice(0, last + 1)
    d_trj = float(np.max(np.abs(traj[sl] - xyz_stored[sl])))
    d_post = float(np.max(np.abs(post_replayed[sl] - post_stored[sl])))
    changed_replayed = np.any(np.abs(post_replayed - pre) > 1e-9, axis=1)
    print("REPLAY_MATCH max|d_cookie_xyz|=%.3e max|d_post_14d|=%.3e steps=%d"
          % (d_trj, d_post, last + 1), flush=True)
    print("changed_stored=%d changed_replayed=%d equal=%s"
          % (int(changed_stored[sl].sum()), int(changed_replayed[sl].sum()),
             bool(np.array_equal(changed_stored[sl], changed_replayed[sl]))), flush=True)
    print("safety_counters " + json.dumps({k: int(st[k]) for k in rule}), flush=True)
    print("count_at_50s " + json.dumps([int(count[i]) for i in range(49, last + 1, 50)]), flush=True)

    mb = mask[sl]
    entry = []
    for i in range(N):
        if not mb[:, i].any():
            continue
        first = int(np.argmax(mb[:, i]))
        entry.append({"cookie": i, "first_in_target_step": first,
                      "in_target_steps": int(mb[:, i].sum()),
                      "in_at_end": bool(mb[last, i]),
                      "finger_contact_steps": int(contact_any[sl, i].sum())})
    print("in_target_cookies " + json.dumps(entry), flush=True)
    lifted = [i for i in range(N) if np.abs(traj[sl, i, 2] - traj[0, i, 2]).max() > 0.003]
    print("cookies_lifted_gt3mm " + json.dumps(sorted(int(i) for i in lifted)), flush=True)
    for i in sorted(args.movers):
        w = mb[:, i]
        mv = np.linalg.norm(np.diff(traj[sl][:, i, :], axis=0), axis=1)
        f = np.linalg.norm(np.diff(eef[sl], axis=0), axis=1)
        rel = traj[sl][:, i, :2] - eef[sl][:, :2]
        drel = np.linalg.norm(np.diff(rel, axis=0), axis=1)
        print("cookie %d first_move=%s max_step_move_mm=%s rigid_with_tcp_steps(drel<2mm)=%s "
              "contact_steps=%s min_dist_tcp_mm=%.1f final_xyz=%s dz_mm=%.1f"
              % (i, int(np.argmax(mv > 0.0005)) if (mv > 0.0005).any() else None,
                 round(1000 * float(mv.max()), 2) if len(mv) else None,
                 int((drel < 0.002).sum()), int(contact_any[sl, i].sum()),
                 1000 * float(np.linalg.norm(rel, axis=1).min()),
                 np.round(traj[last, i], 4).tolist(),
                 1000 * float(traj[last, i, 2] - traj[0, i, 2])), flush=True)

    np.savez_compressed(
        Path(args.out),
        mask=mask[sl], traj=traj[sl], count=count[sl],
        rule_hard=rule["hard"][sl], rule_soft=rule["soft"][sl],
        rule_binfloor=rule["binfloor"][sl], rule_clipped=rule["clipped"][sl],
        contact_any=contact_any[sl], eef_xyz=eef[sl],
        opening=open_arr[sl], applied_opening=app_open_arr[sl], pad_force=force_arr[sl],
        geom=json.dumps(geom), match_max_d_cookie_xyz=d_trj, match_max_d_post=d_post,
    )
    env.close()
    print("WROTE " + str(args.out), flush=True)


if __name__ == "__main__":
    main()
