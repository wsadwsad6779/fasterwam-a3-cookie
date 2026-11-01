"""Stage C0: prove the knobs move, then calibrate the safety tiers.

Three questions this answers before any handover experiment is believable:

1. Does `position_noise_m` actually change the scene? It only does when reset is
   called with `randomize_cookies=True`, which the collector does not do.
2. Does `obs["observation.force"][12:14]` really equal the left finger touch sensors? The
   safety layer reads obs instead of `env._sensor`, which is only safe if so.
3. How much force does the scripted expert itself use? The hard/soft thresholds
   have to sit above that, or the safety layer will abort legitimate grasps.

    python3 calibrate.py --expert-steps 600
"""
import argparse

import numpy as np

import env_factory
import safety_wrap


def scene_fingerprint(env):
    """Explicit truth read, for verification only."""
    return np.round(np.asarray(env.cookie_positions), 5)


def reset_fingerprints(noise, seeds):
    out = {}
    for seed in seeds:
        env = env_factory.make_env(position_noise_m=noise, render_cameras=False)
        try:
            env_factory.reset(env, seed, randomize_cookies=True)
            out[seed] = scene_fingerprint(env)
        finally:
            env.close()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=env_factory.DEFAULT_CONFIG)
    ap.add_argument("--expert-seed", type=int, default=6001)
    ap.add_argument("--expert-steps", type=int, default=600)
    ap.add_argument("--noise", type=float, default=0.003)
    ap.add_argument("--seeds", type=int, nargs="+", default=[11, 21, 22])
    args = ap.parse_args()

    print("== geometry from config (no simulator read) ==")
    for k, v in env_factory.scene_geometry(args.config).items():
        print(f"  {k}: {v}")

    print("\n== force layout ==")
    env = env_factory.make_env(render_cameras=False)
    obs, _ = env_factory.reset(env, args.seeds[0], randomize_cookies=True)
    force = np.asarray(obs[safety_wrap.FORCE_KEY])
    print(f"  obs[{safety_wrap.FORCE_KEY!r}].shape = {force.shape}")
    inner = float(env._sensor("L_finger_inner_touch_sensor")[0])
    outer = float(env._sensor("L_finger_outer_touch_sensor")[0])
    ok = (np.isclose(force[safety_wrap.L_INNER], inner)
          and np.isclose(force[safety_wrap.L_OUTER], outer))
    print(f"  obs[12]={force[safety_wrap.L_INNER]:.4f} vs _sensor inner={inner:.4f}")
    print(f"  obs[13]={force[safety_wrap.L_OUTER]:.4f} vs _sensor outer={outer:.4f}")
    print(f"  layout matches: {ok}")
    env.close()

    print(f"\n== noise sweep (position_noise_m={args.noise}) ==")
    fixed = reset_fingerprints(0.0, args.seeds)
    noisy = reset_fingerprints(args.noise, args.seeds)

    def spread(a, b):
        d = np.abs(np.asarray(a) - np.asarray(b))
        return (1000 * float(np.percentile(d[:, :2], 95)),
                1000 * float(d[:, :2].max()), 1000 * float(d[:, 2].max()))

    # The knob moves each cookie by U(-n, n) in x and y. z is not sampled: it
    # follows from the 50-step settle, so a cookie that lands on a neighbour can
    # differ by centimetres in z while its xy moved by millimetres.
    for seed in args.seeds:
        p95, mx, zmax = spread(noisy[seed], fixed[seed])
        print(f"  seed={seed}: xy p95={p95:.3f} mm max={mx:.3f} mm "
              f"(bound {2 * args.noise * 1000:.1f} mm) | z max={zmax:.3f} mm from settling")
    p95, mx, zmax = spread(noisy[args.seeds[0]], noisy[args.seeds[1]])
    print(f"  seed-to-seed: xy max={mx:.3f} mm | z max={zmax:.3f} mm")

    print("\n== expert force profile (sets the safety tiers) ==")
    from a3_dual_arm_sim.config import load_config
    from a3_dual_arm_sim.controllers.same_column_batch_expert import A3SameColumnBatchExpert
    from a3_dual_arm_sim.tasks.cookie_transfer import (
        A3CookieTransferEnv,
        CookieTransferTaskConfig,
    )

    # Same construction as run_teacher_forcing.py: the success-contract flags do
    # not change the physics, but they must not silently differ either.
    cfg = load_config(args.config)
    task = CookieTransferTaskConfig(
        cookie_count=len(cfg.cookie_transfer.cookie_source_positions_m),
        position_noise_m=0.0,
        require_released=True,
        terminate_on_success=False,
    )
    env = A3CookieTransferEnv(config=cfg, task_config=task,
                              action_mode="cartesian_delta", render_cameras=False)
    obs, _ = env.reset(seed=args.expert_seed, options={"randomize_cookies": True})
    expert = A3SameColumnBatchExpert(env)
    expert.reset()
    peaks, wrist = [], []
    # At reset every force channel is exactly 0, so the layout check above only
    # proves the shape. Re-do it under contact, where the numbers are not zero.
    gaps, contact_steps = [], 0
    for _ in range(args.expert_steps):
        obs, _, terminated, truncated, info = env.step(expert.act(obs))
        f = np.asarray(obs[safety_wrap.FORCE_KEY])
        inner = float(env._sensor("L_finger_inner_touch_sensor")[0])
        outer = float(env._sensor("L_finger_outer_touch_sensor")[0])
        gaps.append(max(abs(f[safety_wrap.L_INNER] - inner),
                        abs(f[safety_wrap.L_OUTER] - outer)))
        if max(inner, outer) > 0.1:
            contact_steps += 1
        peaks.append(float(max(f[safety_wrap.L_INNER], f[safety_wrap.L_OUTER])))
        wrist.append(float(np.linalg.norm(f[0:3])))
        if terminated or truncated or expert.finished:
            break
    env.close()
    peaks = np.asarray(peaks)
    print(f"  steps={len(peaks)} cookies_in_target={info.get('cookies_in_target')} "
          f"failed={expert.failed}")
    print(f"  obs[12:14] vs _sensor under load: max |diff|={max(gaps):.2e} N "
          f"over {contact_steps} contact steps")
    print(f"  pad force p50={np.percentile(peaks, 50):.2f} p95={np.percentile(peaks, 95):.2f} "
          f"max={peaks.max():.2f} N")
    for q in (99, 99.5, 99.9):
        print(f"  p{q}={np.percentile(peaks, q):.2f} N")
    wrist = np.asarray(wrist)
    print(f"  left wrist |f| p50={np.percentile(wrist, 50):.2f} p99={np.percentile(wrist, 99):.2f} "
          f"max={wrist.max():.2f} N")
    # A grasped cookie only ever costs the expert ~1.5 N on the pad sensor, while
    # the same expert calls 8 N a jam (`MAX_INSERTION_FORCE_N`). That leaves a wide
    # band, so the shipped 25/40 N defaults are conservative rather than tuned.
    print(f"  expert peak pad force {peaks.max():.2f} N vs its own jam threshold 8.0 N")
    print(f"  => soft_force_n can sit in [3, 6] N without touching a real grasp; "
          f"keep hard_force_n at the actuator limit (~40 N)")
    print(f"  => the wrist channel reads {np.percentile(wrist, 50):.1f} N while just "
          f"carrying the arm; any press-on-table rule must subtract that baseline")


if __name__ == "__main__":
    main()
