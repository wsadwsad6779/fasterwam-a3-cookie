#!/usr/bin/env python3
"""Real FasterWAM zero-shot inference worker for the A3 cookie task.

Mirrors eval_libero_single.py inference path exactly:
obs -> images/proprio -> model.infer_action -> denorm + gripper chain -> A3 adapter.
"""
import argparse
import inspect
import json
import os
import sys
import time
from pathlib import Path

import imageio
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

BRIDGE_DIR = Path(__file__).resolve().parent
FASTERWAM_ROOT = BRIDGE_DIR.parent / "FasterWAM"
A3_SRC = BRIDGE_DIR.parent / "a3_dual_arm_sim" / "src"

for p in (str(BRIDGE_DIR), str(A3_SRC), str(FASTERWAM_ROOT), str(FASTERWAM_ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

OmegaConf.register_new_resolver("eval", eval)
OmegaConf.register_new_resolver("max", lambda x: max(x))
OmegaConf.register_new_resolver("split", lambda s, idx: s.split("/")[int(idx)])

from a3_dual_arm_sim.tasks.cookie_transfer import A3CookieTransferEnv
from action_adapter_v3 import FasterWAMA3ActionAdapterV3
from vision_adapter_official import prepare_fasterwam_images_official
from observation_adapter import extract_left_eef_state
from gripper_proprio_adapter import a3_opening_to_panda_fingers
from fasterwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor
from fasterwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from fasterwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from fasterwam.utils.pytorch_utils import set_global_seed

CKPT = Path(os.environ.get("FASTERWAM_A3_CKPT",
    str(FASTERWAM_ROOT / "checkpoints/fasterwam_release/libero/step_021700.pt")))
STATS = Path(os.environ.get("FASTERWAM_A3_STATS",
    str(FASTERWAM_ROOT / "checkpoints/fasterwam_release/libero/dataset_stats.json")))
INSTRUCTION = os.environ.get(
    "FASTERWAM_A3_INSTRUCTION",
    "Pick up the cookie with the left robot arm "
    "and transfer it to the right robot arm.")


def load_cfg():
    with initialize_config_dir(version_base="1.3", config_dir=str(FASTERWAM_ROOT / "configs")):
        return compose(config_name="sim_libero.yaml",
                       overrides=["model.redirect_common_files=false"])


def quat_wxyz_to_matrix(quat_wxyz):
    from scipy.spatial.transform import Rotation
    q = np.asarray(quat_wxyz, dtype=np.float64)
    q = q / np.linalg.norm(q)
    xyzw = np.array([q[1], q[2], q[3], q[0]])
    return Rotation.from_quat(xyzw).as_matrix()


def normalize_proprio(proprio, processor):
    state_meta = processor.shape_meta["state"]
    if len(state_meta) != 1:
        raise ValueError("expected single merged state key in shape_meta['state']")
    state_key = state_meta[0]["key"]
    state_batch = {"state": {state_key: torch.as_tensor(proprio, dtype=torch.float32).unsqueeze(0)}}
    state_batch = processor.action_state_transform(state_batch)
    state_batch = processor.normalizer.forward(state_batch)
    return state_batch["state"][state_key]


def denormalize_action(action, processor):
    if action.ndim == 2:
        action = action.unsqueeze(0)
    action_meta = processor.shape_meta["action"]
    if len(action_meta) != 1:
        raise ValueError("expected single merged action key in shape_meta['action']")
    action_key = action_meta[0]["key"]
    normalizer = processor.normalizer.normalizers["action"][action_key]
    action = action.to(dtype=torch.float32, device="cpu")
    return normalizer.backward(action).numpy()


def build_proprio(obs):
    eef6 = extract_left_eef_state(obs)
    opening = float(np.clip(obs["observation.state"][7], 0.0, 1.0))
    fingers = a3_opening_to_panda_fingers(opening)
    return np.concatenate([eef6, fingers]).astype(np.float32)


def predict_action_chunk(obs, model, processor, cfg, seed, model_device):
    rgb, x_np = prepare_fasterwam_images_official(obs)
    image = torch.from_numpy(x_np).to(device=model_device, dtype=model.torch_dtype)
    proprio = normalize_proprio(build_proprio(obs), processor)

    action_horizon = int(cfg.data.train.num_frames) - 1
    prompt = DEFAULT_PROMPT.format(task=INSTRUCTION)

    num_inference_steps_cfg = cfg.EVALUATION.get("num_inference_steps", None)
    num_inference_steps = (
        int(cfg.get("eval_num_inference_steps", 20))
        if num_inference_steps_cfg is None
        else int(num_inference_steps_cfg)
    )

    infer_kwargs = {
        "prompt": prompt,
        "input_image": image,
        "action_horizon": action_horizon,
        "negative_prompt": str(cfg.EVALUATION.get("negative_prompt", "")),
        "text_cfg_scale": float(cfg.EVALUATION.get("text_cfg_scale", 1.0)),
        "num_inference_steps": num_inference_steps,
        "proprio": proprio,
        "sigma_shift": (
            None if cfg.EVALUATION.get("sigma_shift") is None
            else float(cfg.EVALUATION.get("sigma_shift"))
        ),
        "seed": seed,
        "rand_device": str(cfg.EVALUATION.get("rand_device", "cpu")),
        "tiled": bool(cfg.EVALUATION.get("tiled", False)),
    }
    action_infer_mode = str(cfg.EVALUATION.get("action_infer_mode", "joint_rollout")).strip().lower()
    infer_action_fn = model.infer_action
    if action_infer_mode == "one_pass_future_cache":
        infer_action_fn = getattr(model, "infer_action_one_pass_future_cache", None)
        if infer_action_fn is None:
            raise ValueError("model lacks infer_action_one_pass_future_cache")
    num_video_frames = (int(cfg.data.train.num_frames) - 1) // int(cfg.data.train.action_video_freq_ratio) + 1
    if "num_video_frames" in inspect.signature(infer_action_fn).parameters:
        infer_kwargs["num_video_frames"] = num_video_frames

    with torch.no_grad():
        pred = infer_action_fn(**infer_kwargs)

    action = pred["action"]
    action = denormalize_action(action, processor)[0]  # [T, 7]
    if os.environ.get("FASTERWAM_A3_GRIPPER_MODE", "libero01") == "pm1":
        # A3 训练数据夹爪目标就是 ±1（+1=close），min/max 反归一化后直接二值化
        action[..., -1] = np.where(action[..., -1] >= 0.0, 1.0, -1.0)
    else:
        action[..., -1] = action[..., -1] * 2 - 1      # 0/1 -> -1=open, +1=close
        action[..., -1] = action[..., -1] * -1.0       # invert_gripper_action
        if bool(cfg.EVALUATION.get("binarize_gripper", False)):
            action[..., -1] = np.sign(action[..., -1])
    return rgb, action.astype(np.float64)


def run_episode(env, model, processor, cfg, adapter, out_dir, episode_idx, seed, max_steps):
    set_global_seed(int(seed), get_worker_init_fn=False)
    replan_steps = int(cfg.EVALUATION.get("replan_steps", 5))

    obs, info = env.reset(seed=seed)
    pending = []
    chunks_7d = []
    executed_14d = []
    front_frames = []
    merged_frames = []
    current_rgb = None
    t_start = time.time()

    for step in range(max_steps):
        if len(pending) == 0:
            current_rgb, chunk7 = predict_action_chunk(obs, model, processor, cfg, seed, model_device)
            if step == 0:
                print(f"[worker] first chunk shape={chunk7.shape} "
                      f"pos_range=[{chunk7[:, :3].min():.3f},{chunk7[:, :3].max():.3f}] "
                      f"rot_range=[{chunk7[:, 3:6].min():.3f},{chunk7[:, 3:6].max():.3f}] "
                      f"gripper_range=[{chunk7[:, 6].min():.3f},{chunk7[:, 6].max():.3f}]",
                      flush=True)
            chunks_7d.append(chunk7)
            R = quat_wxyz_to_matrix(np.asarray(obs["observation.eef_pose"][3:7], dtype=np.float64))
            pending = [adapter.convert(a, R) for a in chunk7[:replan_steps]]

        a3_action = np.asarray(pending.pop(0))
        obs, reward, terminated, truncated, info = env.step(a3_action)
        executed_14d.append(a3_action)
        front_frames.append(np.asarray(obs["observation.images.front"]).copy())
        merged_frames.append(current_rgb.copy())

        if (step + 1) % 25 == 0:
            print(f"[worker] ep{episode_idx} step={step + 1} "
                  f"cookies={info.get('cookies_in_target', 'N/A')}", flush=True)

        if terminated or truncated:
            break

    duration = time.time() - t_start
    success = bool(info.get("success", False))
    np.savez_compressed(
        out_dir / f"episode_{episode_idx}.npz",
        actions_7d=np.stack(chunks_7d),
        executed_14d=np.asarray(executed_14d),
    )
    imageio.mimsave(out_dir / f"episode_{episode_idx}_front.mp4", front_frames, fps=30)
    imageio.mimsave(out_dir / f"episode_{episode_idx}_model_input.mp4", merged_frames, fps=30)
    return {
        "episode": episode_idx,
        "seed": seed,
        "steps": step + 1,
        "replans": len(chunks_7d),
        "success": success,
        "cookies_in_target": int(info.get("cookies_in_target", -1)),
        "duration_s": round(duration, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--max-steps", type=int, default=400)
    args = ap.parse_args()

    global model_device
    cfg = load_cfg()
    model_device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[worker] device={model_device} torch={torch.__version__}", flush=True)
    print(f"[worker] video_size={list(cfg.data.train.video_size)} "
          f"num_frames={cfg.data.train.num_frames} "
          f"action_video_freq_ratio={cfg.data.train.action_video_freq_ratio}", flush=True)

    model = instantiate(cfg.model, model_dtype=torch.bfloat16, device=model_device)
    model.load_checkpoint(str(CKPT))
    model = model.to(model_device).eval()
    print("[worker] model + checkpoint loaded", flush=True)

    dataset_stats = load_dataset_stats_from_json(str(STATS))
    processor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(dataset_stats)
    print("[worker] processor ready", flush=True)

    out_dir = BRIDGE_DIR / "outputs" / "real_inference" / time.strftime("run_%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[worker] output dir: {out_dir}", flush=True)

    adapter = FasterWAMA3ActionAdapterV3()
    env = A3CookieTransferEnv(action_mode="cartesian_delta", render_cameras=True)

    results = []
    try:
        for ep in range(args.episodes):
            seed = args.seed + ep
            r = run_episode(env, model, processor, cfg, adapter, out_dir, ep, seed, args.max_steps)
            results.append(r)
            print(f"[worker] episode {ep} done: {json.dumps(r)}", flush=True)
    finally:
        env.close()

    with (out_dir / "summary.json").open("w") as f:
        json.dump(results, f, indent=2)
    print(f"[worker] all done, outputs in {out_dir}", flush=True)


if __name__ == "__main__":
    main()
