import argparse
import json
import time

import imageio
import numpy as np
import torch
from hydra.utils import instantiate

import infer_a3_worker as W
import run_guarded as G

CLOSE_M = 0.020
DWELL = 15
COOLDOWN = 40


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-steps", type=int, default=400)
    args = ap.parse_args()

    cfg = W.load_cfg()
    W.model_device = "cuda" if torch.cuda.is_available() else "cpu"
    model = instantiate(cfg.model, model_dtype=torch.bfloat16, device=W.model_device)
    model.load_checkpoint(str(W.CKPT))
    model = model.to(W.model_device).eval()
    stats = W.load_dataset_stats_from_json(str(W.STATS))
    processor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(stats)
    adapter = W.FasterWAMA3ActionAdapterV3()
    print(f"[oracle] model ready device={W.model_device}", flush=True)

    out_dir = W.BRIDGE_DIR / "outputs" / "oracle" / time.strftime("run_%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[oracle] out={out_dir}", flush=True)

    seed = args.seed
    W.set_global_seed(int(seed), get_worker_init_fn=False)
    replan = int(cfg.EVALUATION.get("replan_steps", 5))
    env = G.make_env(0.08)
    obs, info = env.reset(seed=seed)
    c0 = env.cookie_positions.copy()

    pending, chunks, executed = [], [], []
    frames, merged, eef_log, dist_log, open_log, hold_log = [], [], [], [], [], []
    current_rgb = None
    holding, cooldown, triggers, max_lifted = 0, 0, 0, 0
    t0 = time.time()
    step = -1

    for step in range(args.max_steps):
        if not pending:
            current_rgb, chunk7 = W.predict_action_chunk(
                obs, model, processor, cfg, seed, W.model_device)
            chunks.append(chunk7)
            R = W.quat_wxyz_to_matrix(
                np.asarray(obs["observation.eef_pose"][3:7], dtype=np.float64))
            pending = [adapter.convert(a, R) for a in chunk7[:replan]]

        a3 = np.asarray(pending.pop(0), dtype=np.float64).copy()
        dist = G.nearest_cookie_dist(env)
        hold = 0
        if holding > 0:
            a3[:6] = 0.0
            a3[6] = -1.0
            holding -= 1
            hold = 1
        elif cooldown > 0:
            cooldown -= 1
        elif dist < CLOSE_M:
            a3[:6] = 0.0
            a3[6] = -1.0
            holding = DWELL - 1
            cooldown = COOLDOWN
            triggers += 1
            hold = 1

        applied = np.clip(a3, -1.0, 1.0)
        obs, reward, terminated, truncated, info = env.step(applied)
        executed.append(applied)
        eef_log.append(np.asarray(obs["observation.eef_pose"][:3], dtype=np.float64))
        dist_log.append(G.nearest_cookie_dist(env))
        open_log.append(float(obs["observation.state"][7]))
        hold_log.append(hold)
        max_lifted = max(max_lifted,
                         int(np.sum(env.cookie_positions[:, 2] - c0[:, 2] > 0.02)))
        frames.append(np.asarray(obs["observation.images.front"]).copy())
        merged.append(current_rgb.copy())

        if (step + 1) % 25 == 0:
            print(f"[oracle] step={step + 1} cookies={info.get('cookies_in_target')} "
                  f"dist={dist_log[-1]:.4f} open={open_log[-1]:.3f} "
                  f"grasps={triggers} src={info.get('cookies_in_source')}", flush=True)

        if terminated or truncated:
            break

    record = {
        "variant": "oracle_grasp",
        "seed": seed,
        "steps": step + 1,
        "replans": len(chunks),
        "success": bool(info.get("success", False)),
        "cookies_in_target": int(info.get("cookies_in_target", -1)),
        "cookies_in_source": int(info.get("cookies_in_source", -1)),
        "grasp_triggers": int(triggers),
        "max_lifted_gt_2cm": int(max_lifted),
        "min_cookie_dist": round(float(min(dist_log)), 4),
        "duration_s": round(time.time() - t0, 1),
    }
    np.savez_compressed(
        out_dir / f"oracle_episode_{seed}.npz",
        actions_7d=np.stack(chunks),
        executed_14d=np.asarray(executed),
        eef_xyz=np.asarray(eef_log),
        opening=np.asarray(open_log),
        cookie_dist=np.asarray(dist_log),
        hold=np.asarray(hold_log),
    )
    imageio.mimsave(out_dir / f"oracle_episode_{seed}_front.mp4", frames, fps=30)
    imageio.mimsave(out_dir / f"oracle_episode_{seed}_model_input.mp4", merged, fps=30)
    with (out_dir / "summary.json").open("w") as f:
        json.dump([record], f, indent=2)
    print(f"[oracle] done: {json.dumps(record)}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
