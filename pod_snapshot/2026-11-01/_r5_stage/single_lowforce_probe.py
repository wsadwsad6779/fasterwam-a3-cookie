"""Pilot: single-cookie, low-force expert grasps in the a3_fw training scene.

The a3_fw demonstrations were made by A3SameColumnBatchExpert with batch_size 5
and num_batches 2, so every clamp carries a five-cookie bundle. This probe keeps
the same physical grasp machinery but asks it for one cookie per clamp at the
low-force settings the 2x10 expert already uses for single stacks, and dumps the
run in the same npz layout the policy rollouts use so autonomy_metrics.py can
score expert and policy with one scorer.

Nothing here records a dataset; it answers whether the behaviour is achievable.
"""
import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np

from a3_dual_arm_sim.config import load_config
from a3_dual_arm_sim.controllers.same_column_batch_expert import A3SameColumnBatchExpert
from a3_dual_arm_sim.tasks.cookie_transfer import A3CookieTransferEnv, CookieTransferTaskConfig

CONFIG = "/workspace/a3_dual_arm_sim/configs/cookie_batch.yaml"
FORCE_KEY = "observation.force"
L_INNER, L_OUTER = 12, 13


class SingleLowForce(A3SameColumnBatchExpert):
    # batch_size 1 turns the candidate-batch scan into one cookie per clamp;
    # num_batches 10 matches the 2 columns x 5 rows of the training scene.
    batch_size = 1
    num_batches = 10

    @property
    def _is_followup_batch(self):
        # The follow-up batch logic presses the packed neighbours aside before
        # inserting; with one cookie per clamp there is nothing to press aside.
        return False

    def _select_batch(self):
        thickness = 2.0 * float(self.env.COOKIE_HALF_SIZE[1])
        delta = (self.batch_size - 5) * thickness / 0.085
        self.min_grip_opening = max(0.0, 0.28 + delta)
        self.secure_grip_opening = max(0.0, 0.33 + delta)
        self.release_opening = min(1.0, max(0.18, 0.49 + delta))
        self.target_grip_force = 0.45
        self.stable_grip_force = 0.35
        super()._select_batch()

def run_seed(seed, max_steps, noise, out_dir):
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
    env = A3CookieTransferEnv(
        config=cfg, task_config=task, action_mode="cartesian_delta", render_cameras=False
    )
    obs, info = env.reset(seed=int(seed), options={"randomize_cookies": bool(noise)})
    expert = SingleLowForce(env)
    expert.reset()

    xyz, eef, opening, applied, f18, wrist, in_target, names = [], [], [], [], [], [], [], []
    transitions = []
    t0 = time.time()
    stop = None
    for step in range(1, max_steps + 1):
        action = expert.act(obs)
        obs, _, terminated, truncated, info = env.step(action)
        f = np.asarray(obs[FORCE_KEY], dtype=np.float64)
        xyz.append(np.asarray(env.cookie_positions, dtype=np.float64).copy())
        eef.append(np.asarray(obs["observation.eef_pose"][:3], dtype=np.float64).copy())
        opening.append(float(obs["observation.state"][7]))
        applied.append(float(np.asarray(info["applied_action"])[7]))
        f18.append(f.copy())
        wrist.append(f[0:3].copy())
        in_target.append(int(info.get("cookies_in_target", -1)))
        if not names or expert.phase.name != names[-1][1]:
            names.append((step, expert.phase.name))
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

    forces = np.asarray(f18)
    np.savez_compressed(
        out_dir / ("expert_single_%d.npz" % int(seed)),
        cookie_xyz=np.asarray(xyz),
        eef_xyz=np.asarray(eef),
        opening=np.asarray(opening),
        applied_opening=np.asarray(applied),
        pad_force_inner=forces[:, L_INNER],
        pad_force_outer=forces[:, L_OUTER],
        wrist_force=np.asarray(wrist),
        force18=forces,
        changed=np.zeros(len(xyz), dtype=bool),
        in_target=np.asarray(in_target),
        cookie_xyz0=np.asarray(info.get("cookie_positions", xyz[0]), dtype=np.float64),
    )
    summary = {
        "seed": int(seed),
        "noise_m": noise,
        "steps": len(xyz),
        "stop": stop,
        "success": bool(info.get("success", False)),
        "in_target_final": int(info.get("cookies_in_target", -1)),
        "exact_2x5_fill": bool(info.get("exact_2x5_fill", False)),
        "expert_max_pad_force": float(getattr(expert, "max_pad_force", 0.0)),
        "pad_peak_this_run": float(np.max(np.maximum(forces[:, L_INNER], forces[:, L_OUTER]))),
        "batch_reports": json.loads(json.dumps(expert.batch_reports, default=float)),
        "phase_transitions": names,
        "settings": {
            "batch_size": int(SingleLowForce.batch_size),
            "num_batches": int(SingleLowForce.num_batches),
            "target_grip_force": float(expert.target_grip_force),
            "stable_grip_force": float(expert.stable_grip_force),
            "min_grip_opening": float(expert.min_grip_opening),
            "secure_grip_opening": float(expert.secure_grip_opening),
            "release_opening": float(expert.release_opening),
        },
        "wall_s": round(time.time() - t0, 1),
    }
    env.close()
    return summary


def main():
    import shutil

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=[6001])
    ap.add_argument("--max-steps", type=int, default=3200)
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--out", default="/workspace/a3_eval/pilot_single_lowforce")
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists() and any(out.glob("*")):
        raise SystemExit("refusing to write into a non-empty directory: " + str(out))
    free = shutil.disk_usage(str(out.parent if out.parent.exists() else "/workspace")).free
    if free < 1_000_000_000:
        raise SystemExit("only %d bytes free, refusing to start a recording run" % free)
    out.mkdir(parents=True, exist_ok=True)
    print("out=%s free_bytes=%d" % (out, free), flush=True)

    rows = []
    for seed in args.seeds:
        row = run_seed(seed, args.max_steps, args.noise, out)
        rows.append(row)
        print(json.dumps(row, default=float), flush=True)
        (out / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")




def _place_pose_single(self, clearance=0.0, column_index=None):
    # ordered_column() indexes a two-column demonstration order, which the
    # five-per-clamp experts only ever address with batch_index 0 or 1. Ten
    # single-cookie clamps drain the left source column first, so fill the
    # near target column for the first five and the far one for the last five.
    if column_index is None:
        order = getattr(self.env, "demonstration_column_order", (0, 1))
        column_index = order[0] if self.batch_index < 5 else order[1]
    return A3SameColumnBatchExpert._place_pose(self, clearance, column_index)


SingleLowForce._place_pose = _place_pose_single
if __name__ == "__main__":
    main()
