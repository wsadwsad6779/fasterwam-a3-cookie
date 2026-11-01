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

# `load_policy(..., adapter=...)` puts trainable LoRA matrices on top of a
# frozen base, so the instantiated model must contain `lora_*` parameters.
# `sim_libero.yaml` (the release path) has none; the training task config does.
LORA_CAPABLE_CONFIG = ("a3_fasterwam", "a3_2cam")


# LoRA structure used when an adapter is applied at eval time. Mirrors
# `finetune/configs/task/a3_fasterwam.yaml`, the config every A3 run resolves
# from; `--lora-config` overrides it with a real run's `config.yaml`.
DEFAULT_LORA = {"experts": ["video_expert", "action_expert"],
               "target_modules": ["q", "k", "v", "o"],
               "r": 16, "alpha": 32, "dropout": 0.0}


def _import_light_state():
    try:
        from fasterwam import light_state
    except ImportError:
        import light_state
    return light_state


def _adapter_base_matches(meta: dict, base_path: Path) -> bool:
    """True when the adapter records the exact frozen base it was trained on.

    An adapter only means something relative to one base: applying the r4 delta
    to the LIBERO release silently produces a plausible-looking policy that is
    not the trained one, so the recorded provenance is checked, not trusted.
    """
    recorded = meta.get("base_checkpoint")
    if not recorded:
        return False
    try:
        return Path(recorded).resolve() == Path(base_path).resolve()
    except OSError:
        return False


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

    def __init__(self, name, cfg, model, processor, adapter, device, mode="libero01",
                 lora=None, adapter_path=None, adapter_meta=None):
        self.name = name
        self.cfg = cfg
        self.model = model
        self.processor = processor
        self.adapter = adapter
        self.device = device
        self.gripper_mode = mode
        self.lora = dict(lora or {})
        self.adapter_path = str(adapter_path) if adapter_path else None
        self.adapter_meta = dict(adapter_meta or {})
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


def lora_struct_from_config(config_yaml=None):
    """The LoRA structure to wrap the eval model with, read from a real run.

    The adapter file only names tensors, so `r`/`alpha`/`target_modules` have to
    come from somewhere; taking them from the run's resolved `config.yaml` keeps
    one source of truth. Anything that still disagrees is caught by
    `load_adapter(strict=True)`, which compares the key sets.
    """
    from omegaconf import OmegaConf

    path = Path(config_yaml) if config_yaml else None
    if path is None:
        return None
    cfg = OmegaConf.load(str(path))
    block = cfg.get("lora") if hasattr(cfg, "get") else None
    return None if block is None else OmegaConf.to_container(block, resolve=True)


def load_policy(name: str, ckpt=None, stats=None, instruction=None, mode=None,
                adapter=None, lora_config=None) -> Policy:
    import torch
    from hydra.utils import instantiate

    ensure_paths()
    import infer_a3_worker as W

    spec = MODELS[name]
    ckpt = Path(ckpt) if ckpt else spec["ckpt"]
    stats = Path(stats) if stats else spec["stats"]
    adapter = Path(adapter) if adapter else None
    if instruction:
        W.INSTRUCTION = instruction
    for p in (ckpt, stats) + ((adapter,) if adapter is not None else ()):
        if not p.exists():
            raise FileNotFoundError(p)
    if adapter is not None and Path(ckpt).resolve() == Path(spec["ckpt"]).resolve():
        raise SystemExit(
            f"refusing to apply {adapter} to the {name} release base {ckpt}: an "
            "adapter is only meaningful on top of the base it was trained from "
            "(pass that merged checkpoint as --ckpt)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = W.load_cfg()
    model = instantiate(cfg.model, model_dtype=torch.bfloat16, device=device)

    adapter_meta = {}
    lora_struct = None
    if adapter is not None:
        lora_struct = lora_struct_from_config(lora_config) or DEFAULT_LORA
        from fasterwam.lora import apply_lora

        n_layers, n_trainable = apply_lora(model, lora_struct)
        print(f"[policy] applied LoRA r={lora_struct['r']} "
              f"alpha={lora_struct['alpha']} "
              f"layers={n_layers} trainable={n_trainable}", flush=True)

    model.load_checkpoint(str(ckpt))

    if adapter is not None:
        light_state = _import_light_state()
        adapter_meta = light_state.load_adapter(model, str(adapter))
        if not _adapter_base_matches(adapter_meta, ckpt):
            raise SystemExit(
                f"{adapter} was trained on base "
                f"{adapter_meta.get('base_checkpoint')!r}, not {ckpt}; refusing "
                "to evaluate a checkpoint whose provenance does not match")

    model = model.to(device).eval()
    processor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(W.load_dataset_stats_from_json(str(stats)))
    adapter_conv = W.FasterWAMA3ActionAdapterV3()

    mode = gripper_mode(name, mode)
    print(f"[policy] {name} ckpt={ckpt} stats={stats} device={device} "
          f"gripper_mode={mode}", flush=True)
    if adapter is not None:
        print(f"[policy] adapter={adapter} trained_at_step="
              f"{adapter_meta.get('global_step')} base="
              f"{adapter_meta.get('base_checkpoint')}", flush=True)
    print(f"[policy] instruction={W.INSTRUCTION!r}", flush=True)
    return Policy(name, cfg, model, processor, adapter_conv, device, mode,
                  lora=lora_struct, adapter_path=adapter, adapter_meta=adapter_meta)


def describe_policy(name: str, ckpt=None, stats=None, adapter=None) -> dict:
    spec = MODELS[name]
    ckpt = Path(ckpt) if ckpt else spec["ckpt"]
    stats = Path(stats) if stats else spec["stats"]
    out = {"model": name, "checkpoint": fingerprint(ckpt), "stats": fingerprint(stats),
           "normalizer": stats_summary(stats)}
    if adapter:
        out["adapter"] = fingerprint(Path(adapter), nbytes=1 << 30)
        out["adapter_is_the_trainable_part"] = True
        out["merged_checkpoint_written"] = False
    return out


def nearest_source_column_xy(tcp, cfg) -> float:
    """Distance from the tool to the calibrated source grid (no simulator read).

    The stage-B scene is fixed: `cookie_source_positions_m` is the same list the
    task resets from, so this is a calibration constant, not per-episode truth.
    """
    pts = np.asarray(cfg.cookie_transfer.cookie_source_positions_m, dtype=np.float64)
    return float(np.min(np.linalg.norm(pts - np.asarray(tcp)[:2][None, :], axis=1)))
