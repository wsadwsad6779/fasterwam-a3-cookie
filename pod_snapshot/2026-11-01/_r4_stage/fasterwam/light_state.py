"""Adapter-only checkpoints and light training state for the A3 LoRA runs.

A full FasterWAM checkpoint is ~11 GB (5.56 B params in bf16) while the only
trainable parameters are the LoRA matrices (~30.7 M). Saving the whole model
every `save_every` steps is both slow and pointless: the frozen base is the
same file on disk all run long. This module writes what actually changes:

    adapter.pt          LoRA tensors only (bf16, ~61 MB)
    optimizer.pt        AdamW moments, moved to CPU (~245 MB for m=v=fp32)
    scheduler.pt        LR scheduler state
    rng.pt              python / numpy / torch / cuda RNG streams
    trainer_state.json  global_step, epoch, batch_in_epoch, sampler position

`trainer_state.json` is written last, so a crash mid-save leaves no state file
and the resume path refuses to half-restore. Everything is written through a
`.tmp` + `os.replace` pair for the same reason.

Nothing here merges LoRA or touches the frozen base; `adapter.pt` can only be
applied to the exact base it was trained from, which is recorded in its meta.
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import torch

ADAPTER_FORMAT = "fasterwam-lora-adapter-v1"
STATE_FORMAT = "fasterwam-lora-light-state-v1"


def lora_named_parameters(model):
    return [(name, param) for name, param in model.named_parameters() if "lora_" in name]


def adapter_state_dict(model):
    return {name: param.detach().to("cpu", copy=True)
            for name, param in lora_named_parameters(model)}


def save_adapter(model, path, meta=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save({"format": ADAPTER_FORMAT,
                "lora": adapter_state_dict(model),
                "meta": dict(meta or {})}, tmp)
    os.replace(tmp, path)
    return path


def load_adapter(model, path, strict=True, map_location="cpu"):
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if payload.get("format") != ADAPTER_FORMAT:
        raise ValueError(f"{path}: not an adapter checkpoint ({payload.get('format')})")
    adapter = payload["lora"]
    expected = {name for name, _ in lora_named_parameters(model)}
    got = set(adapter)
    if strict and expected != got:
        missing = sorted(expected - got)[:5]
        unexpected = sorted(got - expected)[:5]
        raise RuntimeError(f"{path}: adapter/base mismatch, "
                           f"missing={missing} unexpected={unexpected}")
    # move to each parameter's device/dtype explicitly; load_state_dict would
    # copy into the existing tensors, but bf16 on CPU -> cuda needs the cast.
    params = dict(lora_named_parameters(model))
    with torch.no_grad():
        for name, value in adapter.items():
            param = params[name]
            param.copy_(value.to(device=param.device, dtype=param.dtype))
    return payload.get("meta", {})


def capture_rng():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _optimizer_cpu_state(optimizer):
    inner = getattr(optimizer, "optimizer", optimizer)
    state_dict = inner.state_dict()
    for state in state_dict["state"].values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.detach().to("cpu", copy=True)
    return state_dict


def _optimizer_state_to_device(optimizer, device):
    inner = getattr(optimizer, "optimizer", optimizer)
    for state in inner.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _sampler_state(sampler):
    return {
        "seed": int(getattr(sampler, "seed", 0)),
        "epoch": int(getattr(sampler, "epoch", 0)),
        "epoch_offset": int(getattr(sampler, "epoch_offset", 0)),
        "resume_batch_offset": int(getattr(sampler, "resume_batch_offset", 0)),
        "sample_offset": int(getattr(sampler, "sample_offset", 0)),
        "batch_size": int(getattr(sampler, "batch_size", 0)),
        "num_processes": int(getattr(sampler, "num_processes", 1)),
    }


def _apply_sampler_state(sampler, state):
    sampler.seed = int(state["seed"])
    sampler.epoch = int(state["epoch"])
    sampler.epoch_offset = int(state["epoch_offset"])
    sampler.resume_batch_offset = int(state["resume_batch_offset"])
    if hasattr(sampler, "sample_offset"):
        sampler.sample_offset = int(state.get("sample_offset", 0))
    sampler.batch_size = int(state.get("batch_size", sampler.batch_size))
    sampler.num_processes = int(state.get("num_processes", sampler.num_processes))


def save_light_state(state_dir, *, model, optimizer, scheduler, sampler,
                     global_step, epoch, batch_in_epoch, extra=None,
                     adapter_meta=None):
    """Write adapter + optimizer + scheduler + RNG + counters; returns a manifest."""
    state_dir = Path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)

    def _replace(tmp_name, obj):
        tmp = state_dir / (tmp_name + ".tmp")
        torch.save(obj, tmp)
        os.replace(tmp, state_dir / tmp_name)

    adapter_path = save_adapter(
        model, state_dir / "adapter.pt",
        meta={"global_step": int(global_step), **(adapter_meta or {})})
    _replace("optimizer.pt", _optimizer_cpu_state(optimizer))
    _replace("scheduler.pt", getattr(scheduler, "state_dict",
                                    lambda: {})())
    _replace("rng.pt", capture_rng())

    state = {
        "format": STATE_FORMAT,
        "global_step": int(global_step),
        "epoch": int(epoch),
        "batch_in_epoch": int(batch_in_epoch),
        "sampler": _sampler_state(sampler),
        "extra": dict(extra or {}),
        "files": {"adapter": "adapter.pt", "optimizer": "optimizer.pt",
                  "scheduler": "scheduler.pt", "rng": "rng.pt"},
    }
    tmp = state_dir / "trainer_state.json.tmp"
    tmp.write_text(json.dumps(state, indent=2))
    os.replace(tmp, state_dir / "trainer_state.json")
    return state


def load_light_state(state_dir, *, model, optimizer, scheduler, sampler,
                     map_location="cpu"):
    """Restore everything `save_light_state` wrote; returns the manifest."""
    state_dir = Path(state_dir)
    state_file = state_dir / "trainer_state.json"
    if not state_file.exists():
        raise FileNotFoundError(f"{state_file} missing; refusing a partial restore")
    state = json.loads(state_file.read_text())
    if state.get("format") != STATE_FORMAT:
        raise ValueError(f"{state_dir}: unknown light-state format {state.get('format')}")

    load_adapter(model, state_dir / "adapter.pt", map_location=map_location)

    inner = getattr(optimizer, "optimizer", optimizer)
    inner.load_state_dict(torch.load(state_dir / "optimizer.pt",
                                     map_location=map_location, weights_only=False))
    device = next(p.device for group in inner.param_groups for p in group["params"])
    _optimizer_state_to_device(optimizer, device)

    scheduler_state = torch.load(state_dir / "scheduler.pt",
                                 map_location=map_location, weights_only=False)
    if scheduler_state:
        scheduler.load_state_dict(scheduler_state)

    restore_rng(torch.load(state_dir / "rng.pt", map_location="cpu", weights_only=False))
    _apply_sampler_state(sampler, state["sampler"])
    return state
