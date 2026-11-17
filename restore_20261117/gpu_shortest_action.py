import os
import sys
import json
import time
import traceback
from pathlib import Path

os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = "/workspace/eval_restore_20261117/auxiliary"

import torch
from hydra import initialize_config_dir, compose
from hydra.utils import instantiate
from omegaconf import OmegaConf

import fasterwam.utils.config_resolvers as resolvers
resolvers.register_default_resolvers()

from fasterwam.runtime import _mixed_precision_to_model_dtype, _normalize_mixed_precision

BASE = Path("/workspace/eval_restore_20261117")
CFG_DIR = BASE / "deploy/stage_40k/fw/configs"
CKPT = BASE / "step_040000_merged.pt"
FORCE_CALIB = BASE / "step_040000/force_calib_v1.json"

def main():
    t0 = time.time()
    with initialize_config_dir(config_dir=str(CFG_DIR), version_base="1.3"):
        cfg = compose(
            config_name="train",
            overrides=[
                "task=a3_force_40k",
                f"model.force.norm_stats.file={FORCE_CALIB}",
            ],
        )
    OmegaConf.resolve(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = _mixed_precision_to_model_dtype(_normalize_mixed_precision(cfg.mixed_precision))
    print(f"device={device} dtype={dtype}", flush=True)
    print("instantiating model...", flush=True)
    model = instantiate(cfg.model, model_dtype=dtype, device=device)
    print("loading merged checkpoint...", flush=True)
    model.load_checkpoint(str(CKPT))
    model.eval()
    print("building dummy inputs...", flush=True)
    input_image = torch.zeros((1, 3, 224, 448), dtype=torch.float32)
    proprio = torch.zeros((1, 14), dtype=torch.float32)
    context = torch.zeros((1, 1, 4096), dtype=torch.float32)
    context_mask = torch.ones((1, 1), dtype=torch.bool)
    force_history = torch.zeros((1, 16, 6), dtype=torch.float32)
    force_history_mask = torch.ones((1, 16), dtype=torch.bool)
    print("running infer_action...", flush=True)
    try:
        out = model.infer_action(
            prompt=None,
            input_image=input_image,
            action_horizon=32,
            num_video_frames=17,
            proprio=proprio,
            context=context,
            context_mask=context_mask,
            force_history=force_history,
            force_history_mask=force_history_mask,
            num_inference_steps=5,
            seed=42,
        )
    except Exception as e:
        traceback.print_exc()
        with open(BASE / "gpu_shortest_action_error.json", "w") as f:
            json.dump({"error": str(e), "traceback": traceback.format_exc()}, f)
        raise
    action = out["action"]
    print(f"action shape={tuple(action.shape)} dtype={action.dtype}", flush=True)
    assert action.shape == (32, 7), f"unexpected action shape {action.shape}"
    assert action.dtype == torch.float32
    print("GPU_SHORTEST_ACTION_OK", flush=True)
    print(json.dumps({
        "status": "ok",
        "device": str(device),
        "dtype": str(dtype),
        "action_shape": list(action.shape),
        "action_min": float(action.min()),
        "action_max": float(action.max()),
        "elapsed_sec": round(time.time() - t0, 2),
    }), flush=True)

if __name__ == "__main__":
    main()
