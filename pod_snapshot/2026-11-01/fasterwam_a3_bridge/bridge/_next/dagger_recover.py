"""DAgger-style corrective demonstrations from real C0 failure states.

The r3 C0 rolls are fully determined by the recorded executed actions, so a
fresh, identically seeded environment can be replayed up to any step and then
handed to the scripted expert.  The expert's actions are the training labels;
simulator truth is logged for offline scoring only and never feeds a policy.

    python3 dagger_recover.py --seed 22 --take 90 --replay-only
    python3 dagger_recover.py --seed 22 --take 90 --out outputs/dagger

Outputs per takeover point, under <out>/seed{seed}_s{take}/:

    state_takeover.npz   full physics state (qpos/qvel/ctrl/act/warmstart)
    recover.npz          per-step observations, expert actions, applied
                         actions, phases, cookies_in_target, cookie poses
    *.mp4                front / left_wrist / right_wrist
    report.json          replay fidelity, restore check, recovery verdict
"""
import argparse
import json
from pathlib import Path

import numpy as np

import env_factory
import safety_wrap

R3_DIR = Path("/workspace/fasterwam_a3_bridge/bridge/outputs/r3/C0")
IMAGE_KEYS = ("observation.images.front", "observation.images.left_wrist",
              "observation.images.right_wrist")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--take", type=int, nargs="+", required=True,
                    help="takeover step(s); replay post_14d[:take] first")
    ap.add_argument("--expert-steps", type=int, default=700)
    ap.add_argument("--out", default="outputs/dagger")
    ap.add_argument("--replay-only", action="store_true",
                    help="verify the replay and snapshot the state, no expert")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--dump-frames", action="store_true",
                    help="write frames.dat (expert_steps, 224, 448, 3) uint8: "
                         "front | left_wrist, resized exactly as prepare_a3_arrays.py")
    return ap.parse_args()


IMG = 224


def pack_frame(obs):
    """front and left_wrist side by side, the same tensor the trainer reads."""
    from PIL import Image

    parts = []
    for key in IMAGE_KEYS[:2]:
        img = np.asarray(obs[key])
        if img.shape[0] != IMG or img.shape[1] != IMG:
            img = np.asarray(Image.fromarray(img).resize((IMG, IMG), Image.BILINEAR))
        parts.append(img)
    return np.concatenate(parts, axis=1)


def snapshot(env):
    d = env.data
    return {
        "qpos": d.qpos.copy(),
        "qvel": d.qvel.copy(),
        "ctrl": d.ctrl.copy(),
        "act": d.act.copy(),
        "qacc_warmstart": d.qacc_warmstart.copy(),
        "time": np.float64(d.time),
        "last_applied_action": env.last_applied_action,
        "step_count": np.int64(env._step_count),
        "success_hold_count": np.int64(getattr(env, "_success_hold_count", 0)),
        "source_initially_filled": np.int64(bool(getattr(env, "_source_initially_filled", False))),
        "cookie_positions": np.asarray(env.cookie_positions).copy(),
        "cookie_quats": np.asarray(env.data.xquat[list(env._cookie_bodies)]).copy(),
    }


def restore(env, st):
    import mujoco

    mujoco.mj_resetData(env.model, env.data)
    d = env.data
    for key in ("qpos", "qvel", "ctrl", "act", "qacc_warmstart"):
        getattr(d, key)[:] = st[key]
    d.time = float(st["time"])
    env._last_applied_action = np.asarray(st["last_applied_action"], dtype=np.float64).copy()
    env._step_count = int(st["step_count"])
    if hasattr(env, "_success_hold_count"):
        env._success_hold_count = int(st["success_hold_count"])
    if hasattr(env, "_source_initially_filled"):
        env._source_initially_filled = bool(st["source_initially_filled"])
    mujoco.mj_forward(env.model, env.data)


def make_writers(out_dir, enabled):
    if not enabled:
        return {}
    import imageio

    return {
        key: imageio.get_writer(str(out_dir / f"{key.split('.')[-1]}.mp4"), fps=30)
        for key in IMAGE_KEYS
    }


def close_writers(writers):
    for writer in writers.values():
        writer.close()


def replay_to(env, actions, take, obs=None, info=None):
    """Apply the recorded executed actions and check determinism.

    `take=0` means "take over at the reset state", so the reset observation and
    info have to survive the empty replay.
    """
    eef, opening, applied = [], [], []
    for t in range(take):
        obs, _, terminated, truncated, info = env.step(actions[t])
        eef.append(np.asarray(obs["observation.eef_pose"][:3], dtype=np.float64))
        opening.append(float(obs["observation.state"][7]))
        applied.append(np.asarray(info["applied_action"], dtype=np.float64))
        if terminated or truncated:
            break
    return obs, info, np.asarray(eef), np.asarray(opening), np.asarray(applied)


def run_one(args, ref, take):
    seed = args.seed
    actions = np.asarray(ref["post_14d"], dtype=np.float64)
    out_dir = Path(args.out) / f"seed{seed}_s{take}"
    out_dir.mkdir(parents=True, exist_ok=True)

    env = env_factory.make_env(render_cameras=not args.no_video or args.dump_frames)
    try:
        obs, info = env_factory.reset(env, seed, randomize_cookies=True,
                                      randomize_boxes=False)
        obs, info, eef, opening, applied = replay_to(env, actions, take, obs, info)

        n = len(eef)
        eef_ref = np.asarray(ref["eef_xyz"], dtype=np.float64)[:n]
        open_ref = np.asarray(ref["opening"], dtype=np.float64)[:n]
        applied_ref = np.asarray(ref["applied_canonical"], dtype=np.float64)[:n]
        replay = {
            "steps": int(n),
            "max_abs_eef_diff": float(np.max(np.abs(eef - eef_ref))) if n else None,
            "max_abs_opening_diff": float(np.max(np.abs(opening - open_ref))) if n else None,
            "max_abs_applied_diff": float(np.max(np.abs(applied - applied_ref))) if n else None,
        }

        st = snapshot(env)
        site_before = env.data.site_xpos.copy()
        np.savez_compressed(out_dir / "state_takeover.npz", **st)
        restore(env, st)
        restore_ok = bool(
            np.array_equal(env.data.qpos, st["qpos"])
            and np.array_equal(env.data.qvel, st["qvel"])
            and np.allclose(env.data.site_xpos, site_before, atol=1e-12, rtol=0.0)
            and np.array_equal(env.last_applied_action, st["last_applied_action"])
        )

        start = {
            "cookies_in_target": int(info.get("cookies_in_target", -1)),
            "cookies_in_source": int(info.get("cookies_in_source", -1)),
            "step": int(info.get("step", -1)),
            "eef_z": round(float(obs["observation.eef_pose"][2]), 4),
            "opening": round(float(obs["observation.state"][7]), 4),
        }

        rec = {
            "seed": int(seed),
            "take_step": int(take),
            "replay": replay,
            "state_restore_ok": restore_ok,
            "state_file": str(out_dir / "state_takeover.npz"),
            "start": start,
            "video": not args.no_video,
        }
        if args.replay_only:
            rec["mode"] = "replay_only"
            (out_dir / "report.json").write_text(json.dumps(rec, indent=2))
            print(f"[dagger] {json.dumps(rec)}", flush=True)
            env.close()
            return rec

        from a3_dual_arm_sim.controllers.same_column_batch_expert import (
            A3SameColumnBatchExpert,
        )

        expert = A3SameColumnBatchExpert(env)
        expert.reset()

        writers = make_writers(out_dir, not args.no_video)
        frame_mem = None
        if args.dump_frames:
            frame_mem = np.memmap(out_dir / "frames.dat", dtype=np.uint8, mode="w+",
                                  shape=(args.expert_steps, IMG, IMG * 2, 3))
        act_log, applied_log, eef_log, open_log = [], [], [], []
        force_log, phase_log, phase_steps_log = [], [], []
        cin_log, rew_log, term_log, ncon_log, cookie_log = [], [], [], [], []
        phases = []
        try:
            for _ in range(args.expert_steps):
                # Row t of every state array is the observation the expert acted
                # from. prepare_a3_arrays.py labels action[t] with the delta to
                # state t+1, so the final post-step state is appended below.
                if frame_mem is not None:
                    frame_mem[len(act_log)] = pack_frame(obs)
                action = np.asarray(expert.act(obs), dtype=np.float64).copy()
                act_log.append(action)
                phase = expert.phase.name
                phase_log.append(phase)
                phase_steps_log.append(int(expert.phase_steps))
                eef_log.append(np.asarray(obs["observation.eef_pose"], dtype=np.float64))
                open_log.append(float(obs["observation.state"][7]))
                force_log.append(np.asarray(obs[safety_wrap.FORCE_KEY], dtype=np.float64))
                ncon_log.append(int(env.data.ncon))
                cookie_log.append(np.asarray(env.cookie_positions, dtype=np.float64))
                for key, writer in writers.items():
                    writer.append_data(np.asarray(obs[key]))
                obs, reward, terminated, truncated, info = env.step(action)
                applied_log.append(np.asarray(info["applied_action"], dtype=np.float64))
                cin_log.append(int(info.get("cookies_in_target", -1)))
                rew_log.append(float(reward))
                term_log.append(bool(terminated or truncated))
                if len(act_log) % 50 == 0:
                    print(f"[dagger] seed={seed} s={take} step={len(act_log)} "
                          f"phase={phase} cin={cin_log[-1]}", flush=True)
                if expert.finished or expert.failed or terminated or truncated:
                    break
        finally:
            close_writers(writers)

        if frame_mem is not None:
            frame_mem.flush()
        eef_log.append(np.asarray(obs["observation.eef_pose"], dtype=np.float64))
        open_log.append(float(obs["observation.state"][7]))
        force_log.append(np.asarray(obs[safety_wrap.FORCE_KEY], dtype=np.float64))
        ncon_log.append(int(env.data.ncon))
        cookie_log.append(np.asarray(env.cookie_positions, dtype=np.float64))

        cookie_arr = np.asarray(cookie_log)
        np.savez_compressed(
            out_dir / "recover.npz",
            action14=np.asarray(act_log),
            applied_joint=np.asarray(applied_log),
            eef_pose=np.asarray(eef_log),
            opening=np.asarray(open_log),
            force=np.asarray(force_log),
            phase=np.asarray(phase_log),
            phase_steps=np.asarray(phase_steps_log),
            cookies_in_target=np.asarray(cin_log),
            reward=np.asarray(rew_log),
            done=np.asarray(term_log),
            ncon=np.asarray(ncon_log),
            cookie_xyz=cookie_arr,
        )

        end_count = int(cin_log[-1]) if cin_log else start["cookies_in_target"]
        released = [b for b in getattr(expert, "batch_reports", []) if b.get("released")]
        rec.update({
            "mode": "expert",
            "steps": int(len(act_log)),
            "expert_finished": bool(expert.finished),
            "expert_failed": bool(expert.failed),
            "failure_phase": getattr(expert, "failure_phase", None),
            "failure_reason": getattr(expert, "failure_reason", None),
            "batches_attempted": len(getattr(expert, "batch_reports", [])),
            "batches_released": len(released),
            "batch_reports": getattr(expert, "batch_reports", []),
            "end_cookies_in_target": end_count,
            "delivered": int(end_count - start["cookies_in_target"]),
            "max_pad_force": round(float(np.max(np.abs(np.asarray(force_log)[:, 12:16])))
                                   if force_log else 0.0, 3),
            "phase_counts": {p: phase_log.count(p) for p in sorted(set(phase_log))},
            "recovered": bool((expert.finished and not expert.failed) or len(released) >= 1),
            "npz": str(out_dir / "recover.npz"),
            "state_rows": int(len(eef_log)),
            "action_rows": int(len(act_log)),
            "dump_frames": bool(args.dump_frames),
        })
        (out_dir / "report.json").write_text(json.dumps(rec, indent=2))
        with (Path(args.out) / "summary.jsonl").open("a") as fh:
            fh.write(json.dumps({k: rec[k] for k in (
                "seed", "take_step", "replay", "state_restore_ok", "steps",
                "expert_finished", "expert_failed", "failure_phase",
                "failure_reason", "batches_released", "end_cookies_in_target",
                "delivered", "recovered")}) + "\n")
        print(f"[dagger] {json.dumps(rec)}", flush=True)
        return rec
    finally:
        env.close()


def main():
    args = parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    ref = np.load(R3_DIR / f"lora_C0_{args.seed}.npz")
    for take in args.take:
        run_one(args, ref, take)


if __name__ == "__main__":
    main()
