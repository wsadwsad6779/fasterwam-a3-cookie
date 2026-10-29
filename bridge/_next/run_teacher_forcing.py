"""Stage B1: teacher forcing, plus the action-chain audit it depends on.

The expert drives, so every observation is on the demonstration manifold and the
model's output is never executed. That separates "can this checkpoint fit the A3
data" from "can it stay on track once it acts".

Two measurements, kept apart on purpose.

`chain_audit` (no model, ~seconds). The expert emits a 14-dim cartesian_delta in
A3 units: 1 unit = `cartesian_translation_scale_m` (0.025 m) in world frame,
`cartesian_rotation_scale_rad` (0.10 rad) in the *local* EEF frame, and dim 6 is
a gripper *setpoint* with +1 = fully open. `FasterWAMA3ActionAdapterV3` inverts
exactly that ratio, so the check is: convert the expert's command back to LIBERO
units and compare it with the motion and gripper state the environment actually
produced. If this fails, every model number below is scored against the wrong
target, so it runs first and its own numbers are printed alongside.

`teacher forcing`. Targets are rebuilt from the replayed trajectory with the
training formula, not taken from the expert's action:

    position  (p[t+1] - p[t]) / 0.05            world frame
    rotation  rotvec(R[t+1] R[t]^T) / 0.5       world frame
    gripper   +1 while the measured opening falls, -1 while it rises,
              hold the previous label inside a 1e-4 deadband   (+1 = close)

That is exactly what `prepare_a3_arrays.py` wrote, and `chunk7` is already in
that space, so the comparison needs no adapter and no stale-frame correction.

The gripper channel is the one place where the worker rewrites the model: the
post-processing inside `predict_action_chunk` depends on
`FASTERWAM_A3_GRIPPER_MODE`. Both modes are scored on identical queries, so the
mode whose gripper sign reproduces the label is the correct one and the other is
inverted. `model_pos_rate_*` counts `chunk7[..., 6] > 0` and is deliberately
convention-neutral: read it against `sign_agree_moving`.

    python3 run_teacher_forcing.py --check-only --seeds 7001
    python3 run_teacher_forcing.py --model lora --gripper-modes pm1,libero01 \
        --seeds 7001 7002 7003 --out runs/a3/tf_lora
"""
import argparse
import dataclasses
import json
import os
from pathlib import Path

import numpy as np

LIBERO_POS_M = 0.05
LIBERO_ROT_RAD = 0.5
GRIP_EPS = 1e-4
HORIZONS = (0, 1, 3, 7, 15, 31)
SNAP_KEYS = ("observation.images.front", "observation.images.left_wrist",
             "observation.state", "observation.eef_pose")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["raw", "lora"], default=None)
    ap.add_argument("--check-only", action="store_true",
                    help="run the action-chain audit and stop; no checkpoint")
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--query-stride", type=int, default=20)
    ap.add_argument("--noise-seed", type=int, default=7,
                    help="diffusion seed; fixed across queries, as in run_matrix")
    ap.add_argument("--position-noise", type=float, default=0.0)
    ap.add_argument("--yaw-noise", type=float, default=0.0)
    ap.add_argument("--gripper-modes", default="pm1,libero01")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--stats", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if not args.check_only and args.model is None:
        ap.error("--model is required unless --check-only is set")
    return args


def _rotmats(eef):
    from scipy.spatial.transform import Rotation

    q = np.asarray(eef[:, 3:7], dtype=np.float64)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    xyzw = np.stack([q[:, 1], q[:, 2], q[:, 3], q[:, 0]], axis=1)
    return Rotation.from_quat(xyzw).as_matrix()


def _rotvecs(mats):
    from scipy.spatial.transform import Rotation

    return Rotation.from_matrix(mats).as_rotvec()


def _delta_rotations(R):
    """World-frame R[t+1] R[t]^T as rotation vectors, length T-1."""
    return _rotvecs(np.einsum("tij,tkj->tik", R[1:], R[:-1]))


def _build_env(position_noise, yaw_noise, render=True):
    import dataclasses

    import env_factory
    from a3_dual_arm_sim.config import load_config
    from a3_dual_arm_sim.tasks.cookie_transfer import (
        A3CookieTransferEnv,
        CookieTransferTaskConfig,
    )

    cfg = dataclasses.replace(load_config(env_factory.DEFAULT_CONFIG), horizon=6000)
    task = CookieTransferTaskConfig(
        cookie_count=len(cfg.cookie_transfer.cookie_source_positions_m),
        position_noise_m=float(position_noise),
        yaw_noise_rad=float(yaw_noise),
        # Neither flag changes the physics; they only stop the replay in the
        # middle of a batch if the strict success contract is not met.
        require_released=True,
        terminate_on_success=False,
    )
    env = A3CookieTransferEnv(config=cfg, task_config=task,
                              action_mode="cartesian_delta", render_cameras=render)
    scales = (float(cfg.cartesian_translation_scale_m),
              float(cfg.cartesian_rotation_scale_rad))
    return env, scales


def replay_expert(seed, max_steps, stride, position_noise=0.0, yaw_noise=0.0, render=True):
    from a3_dual_arm_sim.controllers.same_column_batch_expert import (
        A3SameColumnBatchExpert,
    )

    env, scales = _build_env(position_noise, yaw_noise, render)
    try:
        # Noise is only sampled when randomize_cookies is True; at zero noise
        # the two reset options describe the same scene, so this stays on the
        # collector's distribution.
        obs, _ = env.reset(seed=seed, options={"randomize_cookies": True})
        expert = A3SameColumnBatchExpert(env)
        expert.reset()

        snaps, phases, step_ids = [], [], []
        eef, opening, act14 = [], [], []
        info = {}
        for step in range(max_steps):
            if step % stride == 0:
                snaps.append({k: np.array(obs[k], copy=True) for k in SNAP_KEYS})
                phases.append(expert.phase.name)
                step_ids.append(step)
            action = np.asarray(expert.act(obs), dtype=np.float64).copy()
            eef.append(np.asarray(obs["observation.eef_pose"][:7], dtype=np.float64))
            opening.append(float(obs["observation.state"][7]))
            act14.append(action)
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated or expert.finished:
                break
        in_target = int(info.get("cookies_in_target", -1))
    finally:
        env.close()
    return {
        "snaps": snaps, "phases": phases, "step_ids": np.asarray(step_ids),
        "eef": np.asarray(eef), "opening": np.asarray(opening),
        "act14": np.asarray(act14), "scales": scales,
        "cookies_in_target": in_target, "failed": bool(expert.failed),
    }


def build_targets(eef, opening):
    """Training-formula labels in LIBERO units, length T-1."""
    R = _rotmats(eef)
    pos = (eef[1:, :3] - eef[:-1, :3]) / LIBERO_POS_M
    rot = _delta_rotations(R) / LIBERO_ROT_RAD

    d_open = np.diff(opening)
    grip = np.empty(d_open.size)
    prev = -1.0
    for i, d in enumerate(d_open):
        if d < -GRIP_EPS:
            prev = 1.0
        elif d > GRIP_EPS:
            prev = -1.0
        grip[i] = prev
    return np.concatenate([pos, rot, grip[:, None]], axis=1)


def chain_audit(act14, eef, opening, scales):
    """Does the deployed A3 <-> LIBERO convention describe this simulator?"""
    a3_pos, a3_rot = scales
    R = _rotmats(eef)

    pos_cmd = act14[:, :3] * a3_pos
    rot_cmd = np.einsum("tij,tj->ti", R, act14[:, 3:6] * a3_rot)
    pos_err = np.linalg.norm(pos_cmd[:-1] - (eef[1:, :3] - eef[:-1, :3]), axis=1)
    rot_err = np.linalg.norm(rot_cmd[:-1] - _delta_rotations(R), axis=1)

    setpoint = (act14[:, 6] + 1.0) / 2.0
    d_open = np.diff(opening)
    moving = np.abs(d_open) > GRIP_EPS
    aimed = np.abs(setpoint[:-1] - opening[:-1]) > 0.02
    sel = moving & aimed
    agree = (np.sign(setpoint[:-1] - opening[:-1]) == np.sign(d_open))[sel]

    def pct(x, q):
        return round(float(np.percentile(x, q)), 4)

    return {
        "frames": int(len(act14)),
        "a3_translation_scale_m": a3_pos,
        "a3_rotation_scale_rad": a3_rot,
        "pos_err_mm_p50": round(pct(pos_err, 50) * 1000, 3),
        "pos_err_mm_p95": round(pct(pos_err, 95) * 1000, 3),
        "rot_err_mrad_p50": round(pct(rot_err, 50) * 1000, 2),
        "rot_err_mrad_p95": round(pct(rot_err, 95) * 1000, 2),
        "grip_dir_frames": int(sel.sum()),
        "grip_dir_agree_pct": round(100.0 * float(agree.mean()), 2) if sel.any() else None,
        "grip_setpoint_mean": round(float(setpoint.mean()), 3),
        "measured_opening_mean": round(float(opening.mean()), 3),
    }


def evaluate(policy, snaps, step_ids, h_max, noise_seed):
    Q, H = len(snaps), h_max + 1
    chunk7 = np.zeros((Q, H, 7))
    adapted = np.zeros((Q, H, 14))
    valid = np.zeros((Q, H), dtype=bool)
    for q, (snap, t) in enumerate(zip(snaps, step_ids)):
        c7 = np.asarray(policy.chunk(snap, noise_seed))
        ad = np.asarray(policy.adapted(c7, snap))
        take = min(len(c7), H)
        chunk7[q, :take] = c7[:take]
        adapted[q, :take] = ad[:take]
        valid[q, :take] = True
    return chunk7, adapted, valid


def _rate(mask):
    return round(float(mask.mean()), 3) if mask.size else None


def align(targets, valid, opening, step_ids, act14):
    """Per-query target rows plus the expert state at each horizon.

    `cmd_*` is the expert's gripper setpoint, which is what "the expert is
    holding still" has to be judged on: the measured opening rebounds at
    contact, so a still command does not imply a still reading.
    """
    Q, H = valid.shape
    T = len(targets)
    idx = step_ids[:, None] + np.arange(H)[None, :]
    # A query whose horizon reaches past the last full transition has no target;
    # padding it with zeros there would flatter the error.
    valid = valid & (idx < T)

    tgt = np.zeros((Q, H, 7))
    open_now = np.zeros((Q, H))
    open_next = np.zeros((Q, H))
    cmd_now = np.zeros((Q, H))
    cmd_next = np.zeros((Q, H))
    last = len(act14) - 1
    for q in range(Q):
        sel = valid[q]
        tgt[q, sel] = targets[idx[q, sel]]
        open_now[q, sel] = opening[idx[q, sel]]
        open_next[q, sel] = opening[idx[q, sel] + 1]
        cmd_now[q, sel] = act14[idx[q, sel], 6]
        cmd_next[q, sel] = act14[np.minimum(idx[q, sel] + 1, last), 6]
    return tgt, open_now, open_next, cmd_now, cmd_next, valid


def summarize(chunk7, adapted, valid, tgt, open_now, open_next, cmd_now, cmd_next,
              act14, phases, step_ids, scales):
    H = chunk7.shape[1]
    a3_pos, a3_rot = scales

    rows = {}
    for i, h in enumerate(HORIZONS):
        if i >= H:
            break
        sel = valid[:, i]
        if not sel.any():
            continue
        c, t = chunk7[sel, i], tgt[sel, i]
        rows[f"h{h}"] = {
            "n": int(sel.sum()),
            "pos_mae_mm": round(float(np.abs(c[:, :3] - t[:, :3]).mean() * LIBERO_POS_M * 1000), 3),
            "rot_mae_mrad": round(float(np.abs(c[:, 3:6] - t[:, 3:6]).mean() * LIBERO_ROT_RAD * 1000), 2),
            "grip_sign_agree": round(float((np.sign(c[:, 6]) == np.sign(t[:, 6])).mean()), 3),
            "expert_pos_scale_mm": round(float(np.abs(t[:, :3]).mean() * LIBERO_POS_M * 1000), 3),
            "expert_rot_scale_mrad": round(float(np.abs(t[:, 3:6]).mean() * LIBERO_ROT_RAD * 1000), 2),
        }

    by_phase = {}
    for name in sorted(set(phases)):
        sel = np.asarray([p == name for p in phases]) & valid[:, 0]
        if not sel.any():
            continue
        c, t = chunk7[sel, 0], tgt[sel, 0]
        by_phase[name] = {
            "queries": int(sel.sum()),
            "pos_mae_mm": round(float(np.abs(c[:, :3] - t[:, :3]).mean() * LIBERO_POS_M * 1000), 3),
            "rot_mae_mrad": round(float(np.abs(c[:, 3:6] - t[:, 3:6]).mean() * LIBERO_ROT_RAD * 1000), 2),
            "grip_sign_agree": round(float((np.sign(c[:, 6]) == np.sign(t[:, 6])).mean()), 3),
            "expert_grip_label_mean": round(float(t[:, 6].mean()), 3),
            "model_pos_rate": _rate(c[:, 6] > 0),
        }

    sel = valid
    c, t = chunk7[..., 6][sel], tgt[..., 6][sel]
    d_open = (open_next - open_now)[sel]
    now = open_now[sel]
    moving = np.abs(d_open) > GRIP_EPS

    setpoint = (cmd_now + 1.0) / 2.0
    d_sp = (cmd_next - cmd_now)[sel] / 2.0
    setpoint = setpoint[sel]
    cmd_moving = np.abs(d_sp) > GRIP_EPS

    # The measured opening while the pads hold a cookie sits near 0.45-0.51
    # (contact rebound), so the band edges are read off this replay rather than
    # assumed. Frames where the expert holds still are the informative ones:
    # there the label is a held value and the model's bias shows up directly.
    bands = {"open_gt_.85": setpoint > 0.85,
             "mid_.45_.85": (setpoint >= 0.45) & (setpoint <= 0.85),
             "closed_lt_.45": setpoint < 0.45}
    hold_bands = {}
    for name, band in bands.items():
        m = (~cmd_moving) & band
        hold_bands[name] = {"n": int(m.sum()), "model_pos_rate": _rate(c[m] > 0),
                            "label_open_rate": _rate(t[m] < 0)}

    # Live reproduction of the stored-label defect: the setpoint is the expert's
    # real intent, the label came from the measured opening. Same comparison as
    # gripper_label_audit.py, but on this replay instead of the training arrays.
    still = ~cmd_moving
    hold_cmd = still & (setpoint < 0.5)
    open_cmd = still & (setpoint > 0.9)
    label_vs_command = {
        "frames": int(sel.sum()),
        "cmd_moving_pct": round(100.0 * float(cmd_moving.mean()), 2),
        "sign_agree_moving": _rate(np.sign(t[cmd_moving]) == -np.sign(d_sp[cmd_moving])),
        "static_holding_frames": int(hold_cmd.sum()),
        "static_holding_label_open_pct": _rate(t[hold_cmd] < 0),
        "static_open_frames": int(open_cmd.sum()),
        "static_open_label_close_pct": _rate(t[open_cmd] > 0),
    }

    grip = {
        "n": int(sel.sum()),
        "sign_agree_all": round(float((np.sign(c) == np.sign(t)).mean()), 3),
        "sign_agree_moving": _rate(np.sign(c[moving]) == np.sign(t[moving])),
        "expert_moving_pct": round(100.0 * float(moving.mean()), 2),
        "model_pos_rate": _rate(c > 0),
        "model_pos_rate_while_expert_closing": _rate(c[(d_open < -GRIP_EPS)] > 0),
        "model_pos_rate_while_expert_opening": _rate(c[(d_open > GRIP_EPS)] > 0),
        "measured_opening_mean": round(float(now.mean()), 3),
        "hold_bands": hold_bands,
        "label_vs_command": label_vs_command,
    }
    agree = np.sign(chunk7[:, :, 6]) == np.sign(tgt[:, :, 6])
    agree &= valid
    first_bad = np.where(~agree, np.arange(H)[None, :], H).min(axis=1)
    grip["first_disagree_mean_h"] = round(float(first_bad.mean()), 2)

    sel0 = valid[:, 0]
    a = adapted[sel0, 0, :7]
    e = act14[step_ids[sel0]]
    a3_h0 = {
        "n": int(sel0.sum()),
        "pos_mae_mm": round(float(np.abs(a[:, :3] - e[:, :3]).mean() * a3_pos * 1000), 3),
        "rot_mae_mrad": round(float(np.abs(a[:, 3:6] - e[:, 3:6]).mean() * a3_rot * 1000), 2),
        "grip_sign_agree": round(float((np.sign(a[:, 6]) == np.sign(e[:, 6])).mean()), 3),
        "saturated_pct": round(100.0 * float((np.abs(a) > 0.999).mean()), 2),
    }

    return {"horizons": rows, "by_phase": by_phase, "grip": grip, "a3_h0": a3_h0,
            "delta_shape": list(chunk7.shape),
            "valid_queries_pct": round(100.0 * float(valid.mean()), 1)}


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    policy = None
    if not args.check_only:
        from common import load_policy

        policy = load_policy(args.model, args.ckpt, args.stats)
        cfg = policy.cfg
        print(f"[tf] binarize_gripper={cfg.EVALUATION.get('binarize_gripper')} "
              f"replan_steps={cfg.EVALUATION.get('replan_steps')} "
              f"num_frames={cfg.data.train.num_frames}", flush=True)

    modes = [m.strip() for m in args.gripper_modes.split(",") if m.strip()]
    h_max = max(HORIZONS)
    summary = []

    for seed in args.seeds:
        run = replay_expert(seed, args.max_steps, args.query_stride,
                            args.position_noise, args.yaw_noise,
                            render=not args.check_only)
        targets = build_targets(run["eef"], run["opening"])
        chain = chain_audit(run["act14"], run["eef"], run["opening"], run["scales"])
        chain["expert_failed"] = run["failed"]
        print(f"[chain] seed={seed} steps={chain['frames']} "
              f"pos_p50={chain['pos_err_mm_p50']}mm p95={chain['pos_err_mm_p95']}mm "
              f"rot_p50={chain['rot_err_mrad_p50']}mrad "
              f"grip_dir={chain['grip_dir_agree_pct']}% ({chain['grip_dir_frames']})",
              flush=True)

        base = {"seed": int(seed), "chain": chain,
                "expert_steps": len(run["act14"]),
                "expert_cookies_in_target": run["cookies_in_target"],
                "position_noise_m": args.position_noise,
                "yaw_noise_rad": args.yaw_noise}

        if args.check_only:
            summary.append(base)
            continue

        for mode in modes:
            os.environ["FASTERWAM_A3_GRIPPER_MODE"] = mode
            chunk7, adapted, valid = evaluate(policy, run["snaps"], run["step_ids"],
                                              h_max, args.noise_seed)
            tgt, open_now, open_next, cmd_now, cmd_next, valid = align(
                targets, valid, run["opening"], run["step_ids"], run["act14"])
            rec = summarize(chunk7, adapted, valid, tgt, open_now, open_next,
                            cmd_now, cmd_next, run["act14"], run["phases"],
                            run["step_ids"], run["scales"])
            rec.update(base)
            rec.update({"model": args.model, "gripper_mode": mode})
            summary.append(rec)
            np.savez_compressed(
                out_dir / f"tf_{args.model}_{mode}_{seed}.npz",
                chunk7=chunk7, adapted14=adapted, target7=np.where(valid[..., None], tgt, np.nan),
                valid=valid, phase=np.asarray(run["phases"]),
                step_ids=run["step_ids"], target_all=targets,
                opening=run["opening"], eef=run["eef"])
            h0 = rec["horizons"].get("h0", {})
            print(f"[tf] {args.model}/{mode} seed={seed} steps={len(run['act14'])} "
                  f"cookies={run['cookies_in_target']} failed={run['failed']} "
                  f"h0={json.dumps(h0)}", flush=True)
            print(f"[tf] {args.model}/{mode} seed={seed} grip={json.dumps(rec['grip'])}",
                  flush=True)

    name = "chain" if args.check_only else args.model
    (out_dir / f"summary_{name}.json").write_text(json.dumps(summary, indent=2))
    print(f"[tf] done -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
