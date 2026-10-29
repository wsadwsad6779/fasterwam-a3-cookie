"""Stage C0b: is the perturbed scene still a valid scene?

`position_noise_m` is sampled independently per cookie, U(-n, n) in x and y, and
the nominal grid leaves only 0.4 mm between adjacent columns (pitch 50.4 mm vs a
50 mm cookie length) and 2.5 mm between rows (8.833 mm pitch vs a 6.333 mm
width). So the noise is not a small perturbation of a loose scene: at n = 1 mm
two neighbouring columns can be driven into each other by up to 1.6 mm, and at
n > 1.25 mm adjacent rows can too. MuJoCo resolves that penetration with
contacts, which is how a "position noise" experiment silently becomes a
"cookies toppled and shoved" experiment.

This script measures the scene that the reset actually produced (after the
50-step settle), then optionally replays the scripted expert on it, so a low
score can be attributed to the scene or to the expert instead of guessed.

    python3 scene_audit.py --noises 0 0.0005 0.001 0.002 0.003 \
        --seeds 7001 7002 7003 7004 7005 --expert --out outputs/scene_audit
"""
import argparse
import json
from pathlib import Path

import numpy as np

import env_factory


def nominal_grid(config_path):
    from a3_dual_arm_sim.config import load_config

    cfg = load_config(config_path)
    ct = cfg.cookie_transfer
    return cfg, np.asarray(ct.cookie_source_positions_m, dtype=np.float64)


def quat_tilt_deg(quat):
    """Angle between the cookie's own +z and world +z, in degrees."""
    w, x, y, z = np.asarray(quat, dtype=np.float64)
    # Third column of R is the body z axis; for a wxyz quaternion its z
    # component is 1 - 2(x^2 + y^2).
    return float(np.degrees(np.arccos(np.clip(1.0 - 2.0 * (x * x + y * y), -1.0, 1.0))))


def layout_audit(env, cfg, positions, quaternions):
    ct = cfg.cookie_transfer
    hx, hy, _ = (float(v) for v in ct.cookie_half_size_m)
    pos = np.asarray(positions, dtype=np.float64)
    quat = np.asarray(quaternions, dtype=np.float64)
    n = len(pos)

    # Axis-aligned boxes are exact while yaw is zero (the default here). Tilt is
    # reported separately because a tilted cookie's xy footprint is larger than
    # 2hx x 2hy, so a positive gap in this table is not proof of no contact.
    sep = np.full((n, n), np.inf)
    for i in range(n):
        for j in range(i + 1, n):
            dx = abs(pos[i, 0] - pos[j, 0])
            dy = abs(pos[i, 1] - pos[j, 1])
            sep[i, j] = max(dx - 2 * hx, dy - 2 * hy)

    iu = np.triu_indices(n, 1)
    pair_sep = sep[iu]
    worst = np.argsort(pair_sep)[:3]
    worst_pairs = [
        {
            "pair": [int(iu[0][k]), int(iu[1][k])],
            "sep_mm": round(1000.0 * float(pair_sep[k]), 3),
            "dx_mm": round(1000.0 * float(abs(pos[iu[0][k], 0] - pos[iu[1][k], 0])), 3),
            "dy_mm": round(1000.0 * float(abs(pos[iu[0][k], 1] - pos[iu[1][k], 1])), 3),
        }
        for k in worst
    ]

    tilted = [quat_tilt_deg(q) for q in quat]
    z = pos[:, 2]
    z_ref = float(np.median(z))

    try:
        center = np.asarray(env.data.xpos[env._source_bin_body][:2], dtype=np.float64)
        inner = np.asarray(env.SOURCE_INNER_HALF_SIZE, dtype=np.float64)
    except AttributeError:
        center = np.asarray(ct.source_bin_center_m, dtype=np.float64)
        inner = np.asarray(ct.source_bin_half_size_m) - ct.bin_wall_thickness_m
    outside = int(np.sum(
        (np.abs(pos[:, 0] - center[0]) + hx > inner[0])
        | (np.abs(pos[:, 1] - center[1]) + hy > inner[1])
    ))

    return {
        "min_gap_mm": round(1000.0 * float(pair_sep.min()), 3),
        "pairs_penetrating": int(np.sum(pair_sep < -1e-4)),
        "max_penetration_mm": round(1000.0 * float(max(0.0, -pair_sep.min())), 3),
        "worst_pairs": worst_pairs,
        "tilt_max_deg": round(max(tilted), 2),
        "tilted_over_5deg": int(sum(t > 5.0 for t in tilted)),
        "z_max_dev_mm": round(1000.0 * float(np.abs(z - z_ref).max()), 3),
        "outside_source_bin": outside,
    }


def displacement(positions, nominal):
    pos = np.asarray(positions, dtype=np.float64)[:, :2]
    return (1000.0 * (pos - nominal)).tolist()


def run_layout(args, cfg, nominal):
    rows = []
    for noise in args.noises:
        for seed in args.seeds:
            env = env_factory.make_env(config_path=args.config, position_noise_m=noise,
                                       render_cameras=False)
            try:
                env_factory.reset(env, seed, randomize_cookies=True)
                pos = np.asarray(env.cookie_positions, dtype=np.float64)
                quat = np.asarray(env.data.xquat[list(env._cookie_bodies)], dtype=np.float64)
                rec = layout_audit(env, cfg, pos, quat)
            finally:
                env.close()
            rec.update(noise_m=noise, seed=int(seed))
            disp = displacement(pos, nominal)
            rec["max_disp_mm"] = round(max(float(np.hypot(dx, dy)) for dx, dy in disp), 3)
            rec["disp_over_bound"] = int(sum(
                abs(dx) > 1000.0 * noise + 1e-6 or abs(dy) > 1000.0 * noise + 1e-6
                for dx, dy in disp))
            rows.append(rec)
            print(f"  n={noise * 1000:4.1f}mm seed={seed}: "
                  f"min_gap={rec['min_gap_mm']:7.2f}mm "
                  f"pen_pairs={rec['pairs_penetrating']:3d} "
                  f"(max {rec['max_penetration_mm']:.2f}mm) "
                  f"tilt_max={rec['tilt_max_deg']:5.2f}deg ({rec['tilted_over_5deg']:2d}) "
                  f"z_dev={rec['z_max_dev_mm']:6.2f}mm "
                  f"outside={rec['outside_source_bin']:2d} "
                  f"disp_max={rec['max_disp_mm']:5.2f}mm", flush=True)
            if rec["worst_pairs"][0]["sep_mm"] < 0.0:
                print(f"      worst: {rec['worst_pairs']}", flush=True)
    return rows


def run_expert(args):
    from run_teacher_forcing import replay_expert

    rows = []
    for noise in args.noises:
        for seed in args.expert_seeds:
            out = replay_expert(seed, args.max_steps, stride=args.max_steps,
                                position_noise=noise, yaw_noise=args.yaw_noise,
                                render=False)
            rec = {
                "noise_m": noise, "seed": int(seed),
                "steps": int(len(out["eef"])),
                "cookies_in_target": int(out["cookies_in_target"]),
                "failed": bool(out["failed"]),
                "failure_phase": out["failure_phase"],
                "failure_reason": out["failure_reason"],
            }
            rows.append(rec)
            print(f"  n={noise * 1000:4.1f}mm seed={seed}: steps={rec['steps']:3d} "
                  f"cookies={rec['cookies_in_target']:2d} failed={rec['failed']} "
                  f"{rec['failure_phase'] or ''} {rec['failure_reason'] or ''}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=env_factory.DEFAULT_CONFIG)
    ap.add_argument("--noises", type=float, nargs="+",
                    default=[0.0, 0.0005, 0.001, 0.002, 0.003])
    ap.add_argument("--seeds", type=int, nargs="+", default=[7001, 7002, 7003])
    ap.add_argument("--expert", action="store_true")
    ap.add_argument("--expert-seeds", type=int, nargs="+", default=[7001, 7002, 7003])
    ap.add_argument("--max-steps", type=int, default=900)
    ap.add_argument("--yaw-noise", type=float, default=0.0)
    ap.add_argument("--out", default="outputs/scene_audit")
    args = ap.parse_args()

    cfg, nominal = nominal_grid(args.config)
    hx, hy, hz = (float(v) for v in cfg.cookie_transfer.cookie_half_size_m)
    xs = sorted({float(p[0]) for p in nominal})
    ys = sorted({float(p[1]) for p in nominal})
    print(f"cookie half extents: x={1000 * hx:.2f}mm y={1000 * hy:.3f}mm z={1000 * hz:.2f}mm")
    if len(xs) > 1 and len(ys) > 1:
        print(f"nominal clearance: column x={1000 * (xs[1] - xs[0] - 2 * hx):.3f}mm "
              f"row y={1000 * (ys[1] - ys[0] - 2 * hy):.3f}mm")

    print("\n== settled scene after reset(randomize_cookies=True) ==")
    print("   gap is the axis-aligned xy separation after the settle loop; negative = overlap.")
    print("   A cookie tilted more than ~5 deg invalidates the axis-aligned read (its footprint grows).")
    layout = run_layout(args, cfg, nominal)

    expert = []
    if args.expert:
        print("\n== scripted expert on the same scenes ==")
        expert = run_expert(args)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "layout.json").write_text(json.dumps(layout, indent=2))
    if expert:
        (out_dir / "expert.json").write_text(json.dumps(expert, indent=2))

    by_noise = {}
    for rec in expert:
        by_noise.setdefault(rec["noise_m"], []).append(rec)
    if by_noise:
        print("\n== expert summary ==")
        for noise, recs in sorted(by_noise.items()):
            cookies = [r["cookies_in_target"] for r in recs]
            ok = sum(1 for r in recs if r["cookies_in_target"] >= 10)
            print(f"  n={noise * 1000:4.1f}mm: 10/10 on {ok}/{len(recs)} seeds; "
                  f"cookies={cookies}")


if __name__ == "__main__":
    main()
