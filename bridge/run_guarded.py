import argparse
import dataclasses
import json
import time

import imageio
import numpy as np
import torch
from hydra.utils import instantiate

import infer_a3_worker as W
from a3_dual_arm_sim.config import SimConfig
from a3_dual_arm_sim.tasks.cookie_transfer import A3CookieTransferEnv

GUARD_M = 0.030
SOFT_STEP = 0.03
LEFT_GRIP = 6


def make_env(max_gripper_step):
    cfg = dataclasses.replace(SimConfig(horizon=6000), max_gripper_step=max_gripper_step)
    return A3CookieTransferEnv(config=cfg, action_mode="cartesian_delta", render_cameras=True)


def nearest_cookie_dist(env):
    tcp = env.data.site_xpos[env._eef_sites[0]]
    cps = env.cookie_positions
    return float(np.linalg.norm(cps - np.asarray(tcp)[None, :], axis=1).min())


def run_variant(env, model, processor, cfg, adapter, out_dir, name, seed, max_steps, guard_m):
    W.set_global_seed(int(seed), get_worker_init_fn=False)
    replan_steps = int(cfg.EVALUATION.get("replan_steps", 5))
    obs, info = env.reset(seed=seed)
    pending, chunks, executed = [], [], []
    eef_log, open_log, dist_log, guard_log = [], [], [], []
    front_frames, merged_frames = [], []
    current_rgb = None
    overrides = 0
    t0 = time.time()

    for step in range(max_steps):
        if not pending:
            current_rgb, chunk7 = W.predict_action_chunk(
                obs, model, processor, cfg, seed, W.model_device)
            chunks.append(chunk7)
            R = W.quat_wxyz_to_matrix(
                np.asarray(obs["observation.eef_pose"][3:7], dtype=np.float64))
            pending = [adapter.convert(a, R) for a in chunk7[:replan_steps]]

        a3 = np.asarray(pending.pop(0), dtype=np.float64).copy()
        dist_before = nearest_cookie_dist(env)
        override = 0
        if guard_m is not None and a3[LEFT_GRIP] < 0 and dist_before > guard_m:
            a3[LEFT_GRIP] = 1.0
            override = 1
            overrides += 1

        obs, reward, terminated, truncated, info = env.step(a3)
        executed.append(a3)
        eef_log.append(np.asarray(obs["observation.eef_pose"][:3], dtype=np.float64))
        open_log.append(float(obs["observation.state"][7]))
        dist_log.append(nearest_cookie_dist(env))
        guard_log.append(override)
        front_frames.append(np.asarray(obs["observation.images.front"]).copy())
        merged_frames.append(current_rgb.copy())

        if (step + 1) % 50 == 0:
            print(f"[{name}] step={step + 1} cookies={info.get('cookies_in_target')} "
                  f"dist={dist_log[-1]:.4f} open={open_log[-1]:.3f} guard={overrides}",
                  flush=True)

        if terminated or truncated:
            break

    duration = time.time() - t0
    np.savez_compressed(
        out_dir / f"{name}_episode_{seed}.npz",
        actions_7d=np.stack(chunks),
        executed_14d=np.asarray(executed),
        eef_xyz=np.asarray(eef_log),
        opening=np.asarray(open_log),
        cookie_dist=np.asarray(dist_log),
        guard_override=np.asarray(guard_log),
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
        "guard_overrides": int(overrides),
        "min_cookie_dist": round(float(min(dist_log)), 4),
        "duration_s": round(duration, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--variants", default="baseline")
    args = ap.parse_args()

    cfg = W.load_cfg()
    W.model_device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[guard] device={W.model_device}", flush=True)

    model = instantiate(cfg.model, model_dtype=torch.bfloat16, device=W.model_device)
    model.load_checkpoint(str(W.CKPT))
    model = model.to(W.model_device).eval()
    stats = W.load_dataset_stats_from_json(str(W.STATS))
    processor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(stats)
    adapter = W.FasterWAMA3ActionAdapterV3()
    print("[guard] model + processor ready", flush=True)

    out_dir = W.BRIDGE_DIR / "outputs" / "guarded" / time.strftime("run_%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[guard] out={out_dir}", flush=True)

    variants = [
        ("baseline", 0.08, None),
        ("soft", SOFT_STEP, None),
        ("soft_guard", SOFT_STEP, GUARD_M),
    ]
    results = []
    wanted = {v.strip() for v in args.variants.split(",") if v.strip()}
    variants = [v for v in variants if v[0] in wanted]
    for name, gstep, gth in variants:
        env = make_env(gstep)
        try:
            record = run_variant(env, model, processor, cfg, adapter, out_dir,
                                 name, args.seed, args.max_steps, gth)
        finally:
            env.close()
        results.append(record)
        print(f"[guard] {name} done: {json.dumps(record)}", flush=True)
        with (out_dir / "summary.json").open("w") as f:
            json.dump(results, f, indent=2)

    print(f"[guard] all done: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
