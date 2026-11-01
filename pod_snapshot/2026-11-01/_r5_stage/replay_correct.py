"""Correction-demo collection: drive the scripted expert into a real excessive-force
state, then re-plan onto a reachable target and finish a validated low-force grasp.

Nothing here changes the model, the Wrapper, the Safety layer, or the simulator. The
recovery is the expert controller's own re-initialisation (reset plus a new batch
selection) while the same MuJoCo scene continues; the abnormal pre-correction segment
stays inside the recording because that is what the demo is supposed to teach.

Source columns 0 and 1 are reachable for a single-cookie grasp, columns 2 and 3 are
not (demos_single_v30: they always end in expert_failed with 353 of 354 frames above
2 N), so the anomaly is the scene's own, not an injected force.

Reuses /workspace/_r5_stage/replay_forces.py for the environment, the low-force policy,
the phase segmentation and the per-clamp acceptance statistics.
"""
import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np

import replay_forces as R

ANOMALY_PAD_N = 1.6
ANOMALY_FRAMES = 3


class CorrectPress(R.SingleCookieLowForce):
    """Same low-force policy, but allowed to aim at every source column."""

    n_columns = 4

    def _candidate_batches(self):
        source = np.asarray(self.env.SOURCE_POSITIONS)
        columns = sorted(set(source[:, 0].tolist()))[: self.n_columns]
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


def run_correct(seed, variant, max_steps, noise, out_dir, retries=2, recover=(0,)):
    env, obs, info = R.make_env(noise, seed)
    expert = CorrectPress(env)
    expert.variant_index = int(variant)
    expert.reset()
    z0 = np.asarray(env.cookie_positions, dtype=np.float64).copy()
    z_rest = float(np.max(env.cookie_positions[:, 2]))

    phase, pad, xyz, eef, opening, applied, f18, wrist, in_target = [], [], [], [], [], [], [], [], []
    retry_log = []
    attempts = [int(variant)]
    t0, stop = time.time(), None
    step = 0
    while step < max_steps:
        step += 1
        action = expert.act(obs)
        obs, _, terminated, truncated, info = env.step(action)
        f = np.asarray(obs[R.FORCE_KEY], dtype=np.float64)
        phase.append(expert.phase.name)
        pad.append(float(max(f[R.L_INNER], f[R.L_OUTER])))
        xyz.append(np.asarray(env.cookie_positions, dtype=np.float64).copy())
        eef.append(np.asarray(obs["observation.eef_pose"][:3], dtype=np.float64).copy())
        opening.append(float(obs["observation.state"][7]))
        applied.append(float(np.asarray(info["applied_action"])[7]))
        f18.append(f.copy())
        wrist.append(f[0:3].copy())
        in_target.append(int(info.get("cookies_in_target", -1)))
        if terminated:
            stop = "env_safety_stop: " + str(info.get("safety_reason"))
            break
        if truncated:
            stop = "horizon"
            break
        if expert.failed:
            start = len(pad) - 1
            while start > 0 and phase[start - 1] not in R.IDLE_PHASES:
                start -= 1
            window = pad[start:]
            peak = float(max(window)) if window else 0.0
            over = int(sum(1 for v in window if v > ANOMALY_PAD_N))
            kind = ("excessive_pad_force" if peak > ANOMALY_PAD_N and over >= ANOMALY_FRAMES
                    else ("empty_grasp" if peak < 0.3 else "stuck_no_force"))
            retry_log.append({
                "at_step": len(pad),
                "attempt_variant": attempts[-1],
                "reason": str(expert.failure_reason)[:120],
                "peak_pad_n": round(peak, 3),
                "frames_over_1p6n": over,
                "frames_over_2n": int(sum(1 for v in window if v > 2.0)),
                "anomaly_kind": kind,
                "expert_max_pad_force_n": round(float(getattr(expert, "max_pad_force", 0.0)), 3),
            })
            if len(retry_log) > retries:
                stop = "correction_budget"
                break
            variant = recover[min(len(retry_log) - 1, len(recover) - 1)]
            attempts.append(int(variant))
            expert = CorrectPress(env)
            expert.variant_index = variant
            expert.reset()
        if expert.finished:
            stop = "expert_done"
            break

    xyz = np.asarray(xyz)
    eef = np.asarray(eef)
    f18 = np.asarray(f18)
    pad = np.asarray(pad)
    opening = np.asarray(opening)
    np.savez_compressed(
        out_dir / ("expert_correct_%d.npz" % int(seed)),
        cookie_xyz=xyz, cookie_xyz0=z0, eef_xyz=eef, opening=opening,
        applied_opening=np.asarray(applied), pad_force_inner=f18[:, R.L_INNER],
        pad_force_outer=f18[:, R.L_OUTER], wrist_force=np.asarray(wrist), force18=f18,
        changed=np.zeros(len(xyz), dtype=bool), in_target=np.asarray(in_target),
        phase=np.asarray([R.PHASE_NAMES.index(p) for p in phase], dtype=np.int8),
        attempts=np.asarray(attempts, dtype=np.int64),
    )
    clamps = [R.clamp_stats(a, b, phase, pad, xyz, eef, z0, opening, z_rest)
              for a, b in R.segments(phase)]
    rest = xyz[-1]
    inbin = ((np.abs(rest[:, 0] - R.BIN_XY[0]) < R.BIN_INNER_HALF[0])
             & (np.abs(rest[:, 1] - R.BIN_XY[1]) < R.BIN_INNER_HALF[1]))
    moved = np.linalg.norm(rest - z0, axis=1) > 0.05
    upright = np.abs(rest[:, 2] - z0[:, 2]) < 0.004
    anomaly = any(r["frames_over_1p6n"] >= ANOMALY_FRAMES or r["peak_pad_n"] > ANOMALY_PAD_N
                  for r in retry_log)
    last = clamps[-1] if clamps else {}
    recovered = bool(last.get("accept"))
    return {
        "seed": int(seed), "mode": "correct", "noise_m": noise,
        "variant": int(attempts[0]), "attempts": attempts,
        "steps": len(phase), "stop": stop,
        "success": bool(info.get("success", False)),
        "in_target_final": int(np.sum(inbin & moved)),
        "moved_into_bin_upright": int(np.sum(inbin & moved & upright)),
        "moved_into_bin_tilted": int(np.sum(inbin & moved & ~upright)),
        "wrist_p50_n": round(float(np.percentile(np.linalg.norm(np.asarray(wrist), axis=1), 50)), 3),
        "pad_p50_n": round(float(np.percentile(pad, 50)), 3),
        "pad_p95_n": round(float(np.percentile(pad, 95)), 3),
        "pad_max_n": round(float(pad.max()), 3),
        "frames_over_2n": int(np.sum(pad > 2.0)),
        "frames_over_25n": int(np.sum(pad > 25.0)),
        "retry_log": retry_log,
        "anomaly_observed": bool(anomaly),
        "force_anomaly": any(r["anomaly_kind"] == "excessive_pad_force" for r in retry_log),
        "recovered_and_valid": bool(recovered),
        "accept": bool(anomaly and recovered),
        "clamps": clamps,
        "wall_s": round(time.time() - t0, 1),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--variants", type=int, nargs="+", required=True)
    ap.add_argument("--max-steps", type=int, default=1600)
    ap.add_argument("--noise", type=float, default=0.0003)
    ap.add_argument("--retries", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--recover-variants", type=int, nargs="+", default=[0])
    args = ap.parse_args()
    if len(args.seeds) != len(args.variants):
        raise SystemExit("--seeds and --variants must be the same length")

    out = Path(args.out)
    if out.exists() and any(out.glob("*")):
        raise SystemExit("refusing to write into a non-empty directory: " + str(out))
    free = shutil.disk_usage(str(out.parent if out.parent.exists() else "/workspace")).free
    if free < 1_000_000_000:
        raise SystemExit("only %d bytes free, refusing to start" % free)
    out.mkdir(parents=True, exist_ok=True)
    print("out=%s free_bytes=%d driver_sha=%s policy_sha=%s" % (
        out, free,
        hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        hashlib.sha256(Path("/workspace/_r5_stage/replay_forces.py").read_bytes()).hexdigest()),
        flush=True)

    rows = []
    for seed, variant in zip(args.seeds, args.variants):
        row = run_correct(seed, variant, args.max_steps, args.noise, out, args.retries, tuple(args.recover_variants))
        rows.append(row)
        print(json.dumps({k: row[k] for k in ("seed", "variant", "steps", "stop",
                                              "anomaly_observed", "recovered_and_valid",
                                              "accept", "pad_max_n", "frames_over_2n",
                                              "in_target_final")}, default=float), flush=True)
        for r in row["retry_log"]:
            print("      retry", json.dumps(r), flush=True)
        for c in row["clamps"]:
            print("      clamp", json.dumps({k: c.get(k) for k in
                                             ("closure_peak_pad_n", "carry_peak_pad_n",
                                              "post_release_pad_n", "max_cookies_at_grip",
                                              "carried_lift_mm", "placed_in_bin_xy",
                                              "placed_upright", "accept")}, default=float),
                  flush=True)
        (out / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


main()
