"""Shared plumbing for the stage-B/C experiments.

Runs only inside the A3 pod venv, next to `infer_a3_worker.py`. Importing this
module pulls in torch/hydra lazily, so `py_compile` works on any machine.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

# Defaults match the 2026-10-27 cloud layout; the env vars exist so
# `deploy/bootstrap_pod.sh` can honour a non-default WORKSPACE.
BRIDGE_DIR = Path(os.environ.get(
    "FASTERWAM_A3_BRIDGE", "/workspace/fasterwam_a3_bridge/bridge"))
A3_SRC = Path(os.environ.get("FASTERWAM_A3_SRC", "/workspace/a3_dual_arm_sim/src"))
FASTERWAM_ROOT = Path(os.environ.get("FASTERWAM_ROOT", "/workspace/FasterWAM"))
A3_RUNS = Path(os.environ.get("FASTERWAM_A3_RUNS", "/workspace/a3_train_runs"))
A3_DATA = Path(os.environ.get("FASTERWAM_A3_DATA", "/workspace/a3_data"))

# Both checkpoints keep the normalizer they were trained with. The merged LoRA
# was trained against A3 stats (`pretrained_norm_stats` in a3_2cam.yaml); the
# released base was trained against the LIBERO stats shipped beside it. Mixing
# them silently changes the action scale, so each model carries its own file.
#
# The gripper column is what makes this concrete: LIBERO's stats say [0, 1]
# (0/1 labels), A3's say [-1, 1] (the +1=close labels `prepare_a3_arrays.py`
# writes). `predict_action_chunk` therefore needs a different rewrite per model,
# so `gripper_mode` travels with the checkpoint instead of being one global
# env var. Which mode is right is a measurement (stage B1), not a convention
# this file gets to assume.
MODELS = {
    "raw": {
        "ckpt": FASTERWAM_ROOT / "checkpoints/fasterwam_release/libero/step_021700.pt",
        "stats": FASTERWAM_ROOT / "checkpoints/fasterwam_release/libero/dataset_stats.json",
        "gripper_mode": "libero01",
    },
    "lora": {
        "ckpt": A3_RUNS / "a3_lora_r1/step_002000_merged.pt",
        "stats": A3_DATA / "a3_fw/dataset_stats.json",
        "gripper_mode": "pm1",
    },
}


def ensure_paths() -> None:
    for p in (BRIDGE_DIR, A3_SRC, FASTERWAM_ROOT, FASTERWAM_ROOT / "src"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def fingerprint(path, nbytes: int = 1 << 22) -> dict:
    """Cheap identity for an 11 GB checkpoint: size plus head hash."""
    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as f:
        h.update(f.read(nbytes))
    return {"path": str(path), "size": path.stat().st_size, "sha256_head": h.hexdigest()[:16]}


def stats_summary(path) -> dict:
    data = json.loads(Path(path).read_text())
    out = {"num_episodes": data.get("num_episodes"), "num_transition": data.get("num_transition")}
    for key in ("action", "state"):
        block = data.get(key, {}).get("default")
        if block:
            out[key] = {
                "min": np.round(block["global_min"], 4).tolist(),
                "max": np.round(block["global_max"], 4).tolist(),
            }
    return out


class Policy:
    """One checkpoint + its normalizer + the A3 action adapter."""

    def __init__(self, name, cfg, model, processor, adapter, device, mode="libero01"):
        self.name = name
        self.cfg = cfg
        self.model = model
        self.processor = processor
        self.adapter = adapter
        self.device = device
        self.gripper_mode = mode
        self.calls = 0

    def chunk(self, obs, seed: int) -> np.ndarray:
        import os

        import infer_a3_worker as W

        # The worker reads the mode from the environment at call time; this is
        # the only seam it exposes, so the per-checkpoint choice is applied here.
        os.environ["FASTERWAM_A3_GRIPPER_MODE"] = self.gripper_mode
        self.calls += 1
        _, chunk7 = W.predict_action_chunk(obs, self.model, self.processor,
                                           self.cfg, seed, self.device)
        return chunk7

    def adapted(self, chunk7: np.ndarray, obs) -> list:
        import infer_a3_worker as W

        rot = W.quat_wxyz_to_matrix(np.asarray(obs["observation.eef_pose"][3:7], dtype=np.float64))
        return [self.adapter.convert(a, rot) for a in chunk7]


def gripper_mode(name: str, override=None) -> str:
    """Which rewrite `predict_action_chunk` should apply for this checkpoint."""
    return override or MODELS[name].get("gripper_mode", "libero01")


def load_policy(name: str, ckpt=None, stats=None, instruction=None, mode=None) -> Policy:
    import torch
    from hydra.utils import instantiate

    ensure_paths()
    import infer_a3_worker as W

    spec = MODELS[name]
    ckpt = Path(ckpt) if ckpt else spec["ckpt"]
    stats = Path(stats) if stats else spec["stats"]
    if instruction:
        W.INSTRUCTION = instruction
    for p in (ckpt, stats):
        if not p.exists():
            raise FileNotFoundError(p)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = W.load_cfg()
    model = instantiate(cfg.model, model_dtype=torch.bfloat16, device=device)
    model.load_checkpoint(str(ckpt))
    model = model.to(device).eval()
    processor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(W.load_dataset_stats_from_json(str(stats)))
    adapter = W.FasterWAMA3ActionAdapterV3()

    mode = gripper_mode(name, mode)
    print(f"[policy] {name} ckpt={ckpt} stats={stats} device={device} "
          f"gripper_mode={mode}", flush=True)
    print(f"[policy] instruction={W.INSTRUCTION!r}", flush=True)
    return Policy(name, cfg, model, processor, adapter, device, mode)


def describe_policy(name: str, ckpt=None, stats=None) -> dict:
    spec = MODELS[name]
    ckpt = Path(ckpt) if ckpt else spec["ckpt"]
    stats = Path(stats) if stats else spec["stats"]
    return {"model": name, "checkpoint": fingerprint(ckpt), "stats": fingerprint(stats),
            "normalizer": stats_summary(stats)}


def nearest_source_column_xy(tcp, cfg) -> float:
    """Distance from the tool to the calibrated source grid (no simulator read).

    The stage-B scene is fixed: `cookie_source_positions_m` is the same list the
    task resets from, so this is a calibration constant, not per-episode truth.
    """
    pts = np.asarray(cfg.cookie_transfer.cookie_source_positions_m, dtype=np.float64)
    return float(np.min(np.linalg.norm(pts - np.asarray(tcp)[:2][None, :], axis=1)))
