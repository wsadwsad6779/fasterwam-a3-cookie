"""Probe how the observed gripper opening steers the policy's approach.

Loads the worker once, then runs a short rollout per forced opening. The only
intervention is the left-gripper channel; x/y/z/rotation stay the policy's.
Reports TCP height and distance to the nearest cookie so we can pick an
approach opening that the checkpoint actually expects.
"""
import argparse
import dataclasses

import numpy as np
import torch
from hydra.utils import instantiate

import infer_a3_worker as W
from a3_dual_arm_sim.config import SimConfig
from a3_dual_arm_sim.tasks.cookie_transfer import A3CookieTransferEnv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--openings", type=str, default="1.0,0.8,0.6,0.45,0.30")
    ap.add_argument("--replan", type=int, default=5)
    args = ap.parse_args()

    cfg = W.load_cfg()
    W.model_device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[probe] device={W.model_device}", flush=True)
    model = instantiate(cfg.model, model_dtype=torch.bfloat16, device=W.model_device)
    model.load_checkpoint(str(W.CKPT))
    model = model.to(W.model_device).eval()
    stats = W.load_dataset_stats_from_json(str(W.STATS))
    processor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(stats)
    adapter = W.FasterWAMA3ActionAdapterV3()
    print("[probe] model ready", flush=True)

    for g in [float(x) for x in args.openings.split(",") if x.strip()]:
        W.set_global_seed(int(args.seed), get_worker_init_fn=False)
        env = A3CookieTransferEnv(
            config=dataclasses.replace(SimConfig(horizon=6000)),
            action_mode="cartesian_delta", render_cameras=True)
        obs, info = env.reset(seed=args.seed)
        pending, trace = [], []
        for t in range(args.steps):
            if not pending:
                _, chunk7 = W.predict_action_chunk(
                    obs, model, processor, cfg, args.seed, W.model_device)
                R = W.quat_wxyz_to_matrix(
                    np.asarray(obs["observation.eef_pose"][3:7], dtype=np.float64))
                pending = [adapter.convert(a, R) for a in chunk7[:args.replan]]
            a3 = np.asarray(pending.pop(0), dtype=np.float64).copy()
            a3[6] = 2.0 * g - 1.0
            obs, reward, terminated, truncated, info = env.step(a3)
            tcp = np.asarray(obs["observation.eef_pose"][:3], dtype=np.float64)
            d = float(np.linalg.norm(env.cookie_positions - tcp[None, :], axis=1).min())
            if (t + 1) % 30 == 0:
                trace.append("t%d z%.3f d%.3f" % (t + 1, tcp[2], d))
            if terminated or truncated:
                break
        print("[probe] g=%.2f %s" % (g, " ".join(trace)), flush=True)
        env.close()


if __name__ == "__main__":
    main()
