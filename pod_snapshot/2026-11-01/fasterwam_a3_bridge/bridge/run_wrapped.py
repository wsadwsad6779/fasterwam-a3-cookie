"""Closed-loop A3 cookie-transfer evaluation with the optional wrapper.

Loads the worker (model, processor, adapter) from infer_a3_worker, then rolls
out one seed per selected profile from a3_wrap.PROFILES.

Usage (cloud, same dir as infer_a3_worker.py):
    python3 run_wrapped.py --seed 42 --max-steps 400 --variants full
    python3 run_wrapped.py --variants baseline,force,guard,full
"""
import argparse
import dataclasses
import json
import time

import imageio
import numpy as np
import torch
from hydra.utils import instantiate

import a3_wrap
import infer_a3_worker as W
from a3_dual_arm_sim.config import SimConfig, load_config
from a3_dual_arm_sim.tasks.cookie_transfer import A3CookieTransferEnv


def make_env():
    cfg = dataclasses.replace(load_config("/workspace/a3_dual_arm_sim/configs/cookie_batch.yaml"), horizon=6000)
    return A3CookieTransferEnv(config=cfg, action_mode="cartesian_delta", render_cameras=True)


def run_variant(env, model, processor, cfg, adapter, out_dir, name, seed, max_steps, replan):
    W.set_global_seed(int(seed), get_worker_init_fn=False)
    replan_steps = int(replan or cfg.EVALUATION.get("replan_steps", 5))
    prof = a3_wrap.PROFILES[name]
    st = a3_wrap.new_state()
    obs, info = env.reset(seed=seed)
    pending, chunks, executed = [], [], []
    eef_log, open_log, dist_log = [], [], []
    cmd_log, applied_log, hold_log, force_log = [], [], [], []
    phase_log = []
    front_frames, merged_frames = [], []
    current_rgb = None
    t0 = time.time()

    bin_pos = np.asarray(env.data.xpos[env._target_bin_body], dtype=np.float64)
    print(f"[{name}] bin={np.round(bin_pos, 4).tolist()} "
          f"inner_half={np.round(env.TARGET_INNER_HALF_SIZE, 4).tolist()} "
          f"wall={env.TARGET_WALL_HEIGHT:.3f} replan={replan_steps}", flush=True)

    for step in range(max_steps):
        if not pending:
            current_rgb, chunk7 = W.predict_action_chunk(
                obs, model, processor, cfg, seed, W.model_device)
            chunks.append(chunk7)
            R = W.quat_wxyz_to_matrix(
                np.asarray(obs["observation.eef_pose"][3:7], dtype=np.float64))
            pending = [adapter.convert(a, R) for a in chunk7[:replan_steps]]

        a3 = np.asarray(pending.pop(0), dtype=np.float64).copy()
        a3 = a3_wrap.apply(a3, obs, env, st, prof)

        obs, reward, terminated, truncated, info = env.step(a3)
        executed.append(a3)
        tcp = np.asarray(obs["observation.eef_pose"][:3], dtype=np.float64)
        eef_log.append(tcp)
        open_log.append(float(obs["observation.state"][7]))
        applied_log.append(float(info["applied_action"][7]))
        cmd_log.append(float(a3[a3_wrap.LEFT_GRIP]))
        hold_log.append(float(st["hold_steps"]))
        force_log.append((st["inner"], st["outer"]))
        phase_log.append(a3_wrap.PHASES.index(st["phase"]))
        dist_log.append(a3_wrap.nearest_cookie(env, tcp)[0])
        front_frames.append(np.asarray(obs["observation.images.front"]).copy())
        merged_frames.append(current_rgb.copy())

        if (step + 1) % 25 == 0:
            print(f"[{name}] step={step + 1} ph={st['phase']} "
                  f"cookies={info.get('cookies_in_target')} x={tcp[0]:.3f} y={tcp[1]:.3f} z={tcp[2]:.3f} "
                  f"open={open_log[-1]:.3f} appl={applied_log[-1]:.3f} tgt={st['tgt']:.2f} "
                  f"hold={st['hold_steps']} f=({st['inner']:.2f},{st['outer']:.2f}) "
                  f"dist={dist_log[-1]:.4f} bind={st['bin_dist']:.3f} "
                  f"zf={st['zfloor']} st={st['steer']} "
                  f"cyc={st['cycles']} pl={st['placed']}", flush=True)

        if terminated or truncated:
            break

    duration = time.time() - t0
    np.savez_compressed(
        out_dir / f"{name}_episode_{seed}.npz",
        actions_7d=np.stack(chunks),
        executed_14d=np.asarray(executed),
        eef_xyz=np.asarray(eef_log),
        opening=np.asarray(open_log),
        applied_opening=np.asarray(applied_log),
        grip_cmd=np.asarray(cmd_log),
        hold_steps=np.asarray(hold_log),
        pad_force=np.asarray(force_log),
        phase=np.asarray(phase_log),
        cookie_dist=np.asarray(dist_log),
    )
    imageio.mimsave(out_dir / f"{name}_episode_{seed}_front.mp4", front_frames, fps=30)
    imageio.mimsave(out_dir / f"{name}_episode_{seed}_model_input.mp4", merged_frames, fps=30)
    return {
        "variant": name,
        "seed": seed,
        "steps": step + 1,
        "replans": len(chunks),
        "success": bool(info.get("success", False)),
        "cookies_in_target": int(info.get("cookies_in_target", -1)),
        "min_cookie_dist": round(float(min(dist_log)), 4),
        "min_opening": round(float(min(open_log)), 4),
        "min_applied_opening": round(float(min(applied_log)), 4),
        "max_pad_force": round(float(np.max(force_log)), 3),
        "final_phase": st["phase"],
        "hold_steps": int(st["hold_steps"]),
        "released": bool(st["released"]),
        "dropped": bool(st["dropped"]),
        "zfloor_steps": int(st["zfloor"]),
        "place_steer_steps": int(st["steer"]),
        "cycles": int(st["cycles"]),
        "placed": int(st["placed"]),
        "duration_s": round(duration, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--variants", type=str, default="full")
    ap.add_argument("--replan", type=int, default=0)
    args = ap.parse_args()

    cfg = W.load_cfg()
    W.model_device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[wrap] device={W.model_device}", flush=True)

    model = instantiate(cfg.model, model_dtype=torch.bfloat16, device=W.model_device)
    model.load_checkpoint(str(W.CKPT))
    model = model.to(W.model_device).eval()
    stats = W.load_dataset_stats_from_json(str(W.STATS))
    processor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(stats)
    adapter = W.FasterWAMA3ActionAdapterV3()
    print(f"[wrap] ckpt={W.CKPT}", flush=True)
    print(f"[wrap] instruction={W.INSTRUCTION!r}", flush=True)

    out_dir = W.BRIDGE_DIR / "outputs" / "wrapped" / time.strftime("run_%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[wrap] out={out_dir}", flush=True)

    names = [n.strip() for n in args.variants.split(",") if n.strip()]
    results = []
    for name in names:
        if name not in a3_wrap.PROFILES:
            raise SystemExit(f"unknown variant {name}; choose from {list(a3_wrap.PROFILES)}")
        env = make_env()
        try:
            record = run_variant(env, model, processor, cfg, adapter, out_dir,
                                 name, args.seed, args.max_steps, args.replan)
        finally:
            env.close()
        results.append(record)
        print(f"[wrap] {name} done: {json.dumps(record)}", flush=True)
        with (out_dir / "summary.json").open("w") as f:
            json.dump(results, f, indent=2)

    print(f"[wrap] all done: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
