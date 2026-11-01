"""Task 2: replay the expert with force logging and say what a real grasp costs.

Two regimes, one recorder, one scorer:

  bundle -- the settings the a3_fw corpus was made with (cookies per clamp taken
            from the class defaults, the force distribution the r4 policy imitated).
  single -- one cookie per clamp, one clamp per episode, at the low-force settings
            the 2x10 expert uses for single stacks (0.45 N target / 0.35 N stable).

Per clamp it reports the numbers the task asked for: pad force at first contact,
peak while closing, force while holding, peak while carrying, force around the
release, how many cookies actually came with the grip, and whether the cookie is
upright, settled and inside the box on its own at the end.

Nothing here writes a dataset; it fixes the force distribution the demos should use.
"""
import argparse
import dataclasses
import json
import shutil
import time
from pathlib import Path

import numpy as np

from a3_dual_arm_sim.config import load_config
from a3_dual_arm_sim.controllers.phases import CookiePhase
from a3_dual_arm_sim.controllers.same_column_batch_expert import A3SameColumnBatchExpert
from a3_dual_arm_sim.tasks.cookie_transfer import A3CookieTransferEnv, CookieTransferTaskConfig

CONFIG = "/workspace/a3_dual_arm_sim/configs/cookie_batch.yaml"
FORCE_KEY = "observation.force"
L_INNER, L_OUTER = 12, 13
PHASE_NAMES = [p.name for p in CookiePhase]
IDLE_PHASES = ("SUPPORT_BOX", "SELECT_COOKIE", "DONE")
CARRY_PHASES = ("LIFT", "VERIFY_LIFT", "MOVE_TO_SLOT", "DESCEND_TO_PLACE")
CLOSE_PHASES = ("CLOSE", "VERIFY_GRASP")
APPROACH_PHASES = ("APPROACH", "PRE_CLOSE", "ALIGN", "DESCEND")
BIN_XY = np.array([0.095, 0.100])
BIN_INNER_HALF = np.array([0.063 - 0.006, 0.035 - 0.006])
CONTACT_N = 0.05
HOLD_XY_M = 0.025
LIFT_Z_M = 0.02


class SingleCookieLowForce(A3SameColumnBatchExpert):
    batch_size = 1
    num_batches = 1
    variant_index = 0

    def _candidate_batches(self):
        # Spread the collection over source columns and rows: with zero noise every
        # seed would replay the identical column-0 row-0 cookie.
        source = np.asarray(self.env.SOURCE_POSITIONS)
        columns = sorted(set(source[:, 0].tolist()))[:2]
        x = columns[self.variant_index % len(columns)]
        order = sorted(
            np.flatnonzero(np.isclose(source[:, 0], x)).tolist(),
            key=lambda i: source[i, 1],
        )
        start = (self.variant_index // len(columns)) % min(10, len(order))
        for k in range(start, len(order)):
            batch = [order[k]]
            if all(self.env.privileged_cookie_in_source(i) for i in batch):
                yield batch

    @property
    def _is_followup_batch(self):
        return False

    def _select_batch(self):
        thickness = 2.0 * float(self.env.COOKIE_HALF_SIZE[1])
        delta = (self.batch_size - 5) * thickness / 0.085
        self.min_grip_opening = max(0.0, 0.28 + delta)
        self.secure_grip_opening = max(0.0, 0.33 + delta)
        self.release_opening = min(1.0, max(0.18, 0.49 + delta))
        band = 0.45 + 0.05 * (self.variant_index % 3)
        self.target_grip_force = band
        self.stable_grip_force = band - 0.10
        super()._select_batch()

    def _place_pose(self, clearance=0.0, column_index=None):
        # One cookie per clamp gets its own row. The target grid puts 6.33 mm
        # between rows and the slab is 6.34 mm thick, so releasing every cookie
        # at the column mean row leans the last ones against a neighbour.
        if column_index is None:
            order = getattr(self.env, "demonstration_column_order", (0, 1))
            column_index = order[self.variant_index % len(order)]
        position, rotation = A3SameColumnBatchExpert._place_pose(self, clearance, column_index)
        slots = np.asarray(self.env.TARGET_SLOTS_LOCAL)
        column_x = np.unique(slots[:, 0])[column_index]
        rows = np.sort(slots[np.isclose(slots[:, 0], column_x), 1])
        offset = rows[(self.variant_index // 2) % len(rows)] - rows.mean()
        box_r = self.data.xmat[self._tb_id].reshape(3, 3)
        return position + box_r @ np.array([0.0, offset, 0.0]), rotation


def make_env(noise, seed):
    cfg = dataclasses.replace(load_config(CONFIG), horizon=6000)
    ct = cfg.cookie_transfer
    task = CookieTransferTaskConfig(
        cookie_count=len(ct.cookie_source_positions_m),
        position_noise_m=noise,
        yaw_noise_rad=0.015 if noise else 0.0,
        require_exact_slots=False,
        require_released=True,
        terminate_on_success=False,
    )
    env = A3CookieTransferEnv(config=cfg, task_config=task, action_mode="cartesian_delta",
                              render_cameras=False)
    obs, info = env.reset(seed=int(seed), options={"randomize_cookies": bool(noise)})
    return env, obs, info


def segments(phase):
    out, start = [], None
    for i, name in enumerate(phase):
        busy = name not in IDLE_PHASES
        if busy and start is None:
            start = i
        elif not busy and start is not None:
            out.append((start, i))
            start = None
    if start is not None:
        out.append((start, len(phase)))
    return out


def clamp_stats(a, b, phase, pad, xyz, eef, z0, opening, carried_rest_z):
    idx = range(a, b)
    win = lambda names: [i for i in idx if phase[i] in names]
    ap, cl, cy = win(APPROACH_PHASES), win(CLOSE_PHASES), win(CARRY_PHASES)
    rel = win(("OPEN", "VERIFY_RELEASE", "RETRACT"))
    rise = xyz[:, :, 2] - z0[:, 2]
    peak_rise = rise[a:b].max(axis=0)
    travel = np.linalg.norm(xyz[a:b, :, :2] - z0[:, :2], axis=2).max(axis=0)
    top = int(np.argmax(peak_rise))
    stats = {"steps": b - a, "phases": sorted({phase[i] for i in idx})}
    touched = [i for i in idx if pad[i] > CONTACT_N]
    stats["first_contact_pad_n"] = round(float(pad[touched[0]]), 3) if touched else None
    stats["closure_peak_pad_n"] = round(float(max((pad[i] for i in cl), default=0.0)), 3)
    stats["hold_mean_pad_n"] = round(float(np.mean([pad[i] for i in cy])), 3) if cy else None
    stats["carry_peak_pad_n"] = round(float(max((pad[i] for i in cy), default=0.0)), 3)
    if rel:
        stats["pre_release_pad_n"] = round(float(max(pad[max(rel[0] - 10, 0):rel[0] + 1])), 3)
        stats["post_release_pad_n"] = round(float(np.mean(pad[rel[0]:min(rel[-1] + 11, b)])), 3)
    n_at_grip = [int(np.sum((np.linalg.norm(xyz[i, :, :2] - eef[i, :2], axis=1) < HOLD_XY_M)
                            & (rise[i] > LIFT_Z_M))) for i in cy]
    stats["max_cookies_at_grip"] = int(max(n_at_grip, default=0))
    stats["cookies_raised_20mm"] = int((peak_rise > 0.020).sum())
    stats["min_opening"] = round(float(min(opening[a:b])), 4)
    stats["max_opening"] = round(float(max(opening[a:b])), 4)
    stats["carried_index"] = top
    stats["carried_lift_mm"] = round(float(peak_rise[top]) * 1000, 1)
    stats["carried_distance_mm"] = round(float(travel[top]) * 1000, 1)
    end = xyz[b - 1, top]
    stats["placed_in_bin_xy"] = bool(abs(end[0] - BIN_XY[0]) < BIN_INNER_HALF[0]
                                     and abs(end[1] - BIN_XY[1]) < BIN_INNER_HALF[1])
    stats["placed_upright"] = bool(abs(end[2] - z0[top, 2]) < 0.004)
    stats["accept"] = bool(stats["max_cookies_at_grip"] == 1
                           and stats["carried_lift_mm"] >= 60.0
                           and (stats["carry_peak_pad_n"] or 0.0) <= 2.0
                           and stats["placed_in_bin_xy"]
                           and stats["placed_upright"])
    return stats


def run_episode(seed, mode, max_steps, noise, out_dir, clamp_limit=0, variant=0):
    env, obs, info = make_env(noise, seed)
    expert = (SingleCookieLowForce(env) if mode == "single" else A3SameColumnBatchExpert(env))
    if clamp_limit:
        expert.num_batches = int(clamp_limit)
    expert.variant_index = int(variant)
    expert.reset()
    z0_live = np.asarray(env.cookie_positions, dtype=np.float64).copy()
    z_rest = float(np.max(env.cookie_positions[:, 2]))

    phase, pad, xyz, eef, opening, applied, f18, wrist, in_target = [], [], [], [], [], [], [], [], []
    t0, stop = time.time(), None
    for step in range(1, max_steps + 1):
        action = expert.act(obs)
        obs, _, terminated, truncated, info = env.step(action)
        f = np.asarray(obs[FORCE_KEY], dtype=np.float64)
        phase.append(expert.phase.name)
        pad.append(float(max(f[L_INNER], f[L_OUTER])))
        xyz.append(np.asarray(env.cookie_positions, dtype=np.float64).copy())
        eef.append(np.asarray(obs["observation.eef_pose"][:3], dtype=np.float64).copy())
        opening.append(float(obs["observation.state"][7]))
        applied.append(float(np.asarray(info["applied_action"])[7]))
        f18.append(f.copy())
        wrist.append(f[0:3].copy())
        in_target.append(int(info.get("cookies_in_target", -1)))
        if terminated:
            stop = "env_safety_stop: " + str(info.get("safety_reason"))
        elif truncated:
            stop = "horizon"
        elif expert.failed:
            stop = "expert_failed: " + str(expert.failure_reason)
        elif expert.finished:
            stop = "expert_done"
        if stop:
            break

    xyz = np.asarray(xyz); eef = np.asarray(eef); f18 = np.asarray(f18)
    pad = np.asarray(pad); opening = np.asarray(opening)
    z0 = z0_live
    np.savez_compressed(
        out_dir / ("expert_%s_%d.npz" % (mode, int(seed))),
        cookie_xyz=xyz, cookie_xyz0=z0, eef_xyz=eef, opening=opening,
        applied_opening=np.asarray(applied), pad_force_inner=f18[:, L_INNER],
        pad_force_outer=f18[:, L_OUTER], wrist_force=np.asarray(wrist), force18=f18,
        changed=np.zeros(len(xyz), dtype=bool), in_target=np.asarray(in_target),
        phase=np.asarray([PHASE_NAMES.index(p) for p in phase], dtype=np.int8),
    )
    clamps = [clamp_stats(a, b, phase, pad, xyz, eef, z0, opening, z_rest)
              for a, b in segments(phase)]
    rest = xyz[-1]
    inbin = ((np.abs(rest[:, 0] - BIN_XY[0]) < BIN_INNER_HALF[0])
             & (np.abs(rest[:, 1] - BIN_XY[1]) < BIN_INNER_HALF[1]))
    settled = np.abs(rest[:, 2] - z0[:, 2]) < 0.004
    moved = np.linalg.norm(rest - z0, axis=1) > 0.05
    return {
        "seed": int(seed), "mode": mode, "noise_m": noise, "steps": len(phase),
        "stop": stop, "success": bool(info.get("success", False)),
        "in_target_final": int(info.get("cookies_in_target", -1)),
        "rest_z_nominal_mm": round(z_rest * 1000, 1),
        "moved_into_bin_upright": int(np.sum(inbin & settled & moved)),
        "moved_into_bin_tilted": int(np.sum(inbin & ~settled & moved)),
        "wrist_p50_n": round(float(np.percentile(np.linalg.norm(np.asarray(wrist), axis=1), 50)), 2),
        "pad_p50_n": round(float(np.percentile(pad, 50)), 3),
        "pad_p95_n": round(float(np.percentile(pad, 95)), 3),
        "pad_max_n": round(float(pad.max()), 3),
        "frames_over_2n": int((pad > 2).sum()), "frames_over_25n": int((pad > 25).sum()),
        "clamps": clamps,
        "batch_reports": json.loads(json.dumps(expert.batch_reports, default=float)),
        "settings": {"batch_size": int(expert.batch_size), "num_batches": int(expert.num_batches),
                     "target_grip_force": float(expert.target_grip_force),
                     "stable_grip_force": float(expert.stable_grip_force),
                     "min_grip_opening": float(expert.min_grip_opening),
                     "secure_grip_opening": float(expert.secure_grip_opening),
                     "release_opening": float(expert.release_opening)},
        "wall_s": round(time.time() - t0, 1),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", required=True, choices=["single", "bundle"])
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--max-steps", type=int, default=1600)
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--clamps", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists() and any(out.glob("*")):
        raise SystemExit("refusing to write into a non-empty directory: " + str(out))
    free = shutil.disk_usage(str(out.parent if out.parent.exists() else "/workspace")).free
    if free < 1_000_000_000:
        raise SystemExit("only %d bytes free, refusing to start" % free)
    out.mkdir(parents=True, exist_ok=True)
    print("out=%s free_bytes=%d" % (out, free), flush=True)

    rows = []
    for index, seed in enumerate(args.seeds):
        row = run_episode(seed, args.mode, args.max_steps, args.noise, out,
                          args.clamps, variant=index)
        rows.append(row)
        print(json.dumps({k: row[k] for k in ("seed", "mode", "steps", "stop",
                                              "in_target_final", "moved_into_bin_upright",
                                              "moved_into_bin_tilted", "pad_max_n",
                                              "frames_over_2n")}, default=float), flush=True)
        for c in row["clamps"]:
            print("     ", json.dumps(c, default=float), flush=True)
        (out / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

