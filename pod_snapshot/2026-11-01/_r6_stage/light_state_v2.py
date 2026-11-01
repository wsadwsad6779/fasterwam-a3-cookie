"""Independent test branch for light-state completeness.

Nothing here is imported by training or evaluation, and no r1..r5 checkpoint is
read for writing: the live module stays at
`/workspace/FasterWAM/src/fasterwam/light_state.py` and this file only wraps it.

Why it exists: `lora_named_parameters()` keeps only names containing "lora_",
but the trainer also optimises `proprio_encoder` (`trainer.py:86-88`), so every
light state silently drops 2 of the 786 trainable tensors while `optimizer.pt`
still carries their Adam moments. A v2 adapter carries every trainable tensor
and refuses to pretend a legacy file is complete.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch

from fasterwam import light_state as L

ADAPTER_FORMAT_V2 = "fasterwam-lora-adapter-v2"
STATE_FORMAT_V2 = "fasterwam-lora-light-state-v2"


def trainable_named_parameters(model):
    return [(n, p) for n, p in model.named_parameters() if p.requires_grad]


def coverage(model):
    """What v1 keeps, what v1 drops, and what the optimiser is actually stepping."""
    trainable = trainable_named_parameters(model)
    lora = [(n, p) for n, p in trainable if "lora_" in n]
    extra = [(n, p) for n, p in trainable if "lora_" not in n]
    return {
        "trainable_tensors": len(trainable),
        "trainable_elements": sum(int(p.numel()) for _, p in trainable),
        "kept_by_v1": len(lora),
        "kept_by_v1_elements": sum(int(p.numel()) for _, p in lora),
        "dropped_by_v1": [{"name": n, "shape": list(p.shape), "elements": int(p.numel())}
                          for n, p in extra],
        "dropped_by_v1_elements": sum(int(p.numel()) for _, p in extra),
    }


def _split_trainable(model):
    lora, extra = {}, {}
    for name, param in trainable_named_parameters(model):
        held = param.detach().to("cpu", copy=True)
        (lora if "lora_" in name else extra)[name] = held
    return lora, extra


def save_adapter_v2(model, path, meta=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lora, extra = _split_trainable(model)
    tmp = path.with_name(path.name + ".tmp")
    torch.save({"format": ADAPTER_FORMAT_V2,
                "lora": lora,
                "extra_trainable": extra,
                "meta": dict(meta or {})}, tmp)
    os.replace(tmp, path)
    return path


def _copy_into(named, blob, path):
    unknown = sorted(set(blob) - set(named))
    if unknown:
        raise RuntimeError("%s: adapter has tensors the model does not: %s"
                           % (path, unknown[:5]))
    shapes = ["%s: file%s model%s" % (n, list(blob[n].shape), list(named[n].shape))
              for n in blob if list(blob[n].shape) != list(named[n].shape)]
    if shapes:
        raise RuntimeError("%s: shape mismatch: %s" % (path, shapes[:5]))
    with torch.no_grad():
        for name, value in blob.items():
            param = named[name]
            param.copy_(value.to(device=param.device, dtype=param.dtype))


def load_adapter_v2(model, path, map_location="cpu", allow_legacy=False):
    """Restore every trainable tensor. A v1 file is incomplete and says so.

    Matching runs against `model.named_parameters()` rather than the current
    trainable set, so a model built for inference (where `requires_grad` may be
    switched off) still loads; the completeness check is the other direction --
    anything trainable now but absent from the file is reported.
    """
    path = Path(path)
    payload = torch.load(path, map_location=map_location, weights_only=False)
    fmt = payload.get("format")
    named = dict(model.named_parameters())
    trainable_names = {n for n, _ in trainable_named_parameters(model)}

    if fmt == L.ADAPTER_FORMAT:
        blob = dict(payload["lora"])
        missing = sorted(trainable_names - set(blob))
        if missing and not allow_legacy:
            raise RuntimeError(
                "%s: v1 adapter covers %d of %d trainable tensors. These were "
                "trainable but never saved: %s. The file cannot represent the "
                "trained model; it only restores the LoRA part and leaves the "
                "rest at whatever the base checkpoint supplied. Pass "
                "allow_legacy=True to load it anyway."
                % (path, len(blob), len(trainable_names), missing))
        _copy_into(named, blob, path)
        return {"format": "v1", "restored_tensors": len(blob),
                "trainable_tensors": len(trainable_names),
                "not_restored": missing, "meta": payload.get("meta", {})}

    if fmt != ADAPTER_FORMAT_V2:
        raise ValueError("%s: not a recognised adapter checkpoint (%s)" % (path, fmt))

    blob = dict(payload["lora"])
    blob.update(payload.get("extra_trainable", {}))
    missing = sorted(trainable_names - set(blob))
    if missing:
        raise RuntimeError("%s: v2 adapter is missing trainable tensors %s"
                           % (path, missing[:5]))
    _copy_into(named, blob, path)
    return {"format": "v2", "restored_tensors": len(blob),
            "trainable_tensors": len(trainable_names),
            "not_restored": [], "meta": payload.get("meta", {})}


def save_light_state_v2(state_dir, *, model, optimizer, scheduler, sampler,
                        global_step, epoch, batch_in_epoch, extra=None,
                        adapter_meta=None):
    """Same discipline as `save_light_state`: tmp+replace, state file last."""
    state_dir = Path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)

    def _replace(tmp_name, obj):
        tmp = state_dir / (tmp_name + ".tmp")
        torch.save(obj, tmp)
        os.replace(tmp, state_dir / tmp_name)

    cov = coverage(model)
    save_adapter_v2(model, state_dir / "adapter.pt",
                    meta={"global_step": int(global_step),
                          "trainable_tensors": cov["trainable_tensors"],
                          "dropped_by_v1": [d["name"] for d in cov["dropped_by_v1"]],
                          **(adapter_meta or {})})
    _replace("optimizer.pt", L._optimizer_cpu_state(optimizer))
    _replace("scheduler.pt", getattr(scheduler, "state_dict", lambda: {})())
    _replace("rng.pt", L.capture_rng())

    state = {"format": STATE_FORMAT_V2,
             "global_step": int(global_step),
             "epoch": int(epoch),
             "batch_in_epoch": int(batch_in_epoch),
             "sampler": L._sampler_state(sampler),
             "extra": dict(extra or {}),
             "coverage": cov,
             "files": {"adapter": "adapter.pt", "optimizer": "optimizer.pt",
                       "scheduler": "scheduler.pt", "rng": "rng.pt"}}
    tmp = state_dir / "trainer_state.json.tmp"
    tmp.write_text(json.dumps(state, indent=2))
    os.replace(tmp, state_dir / "trainer_state.json")
    return state


def load_light_state_v2(state_dir, *, model, optimizer, scheduler, sampler,
                        map_location="cpu", allow_legacy=False):
    state_dir = Path(state_dir)
    state_file = state_dir / "trainer_state.json"
    if not state_file.exists():
        raise FileNotFoundError("%s missing; refusing a partial restore" % state_file)
    state = json.loads(state_file.read_text())
    if state.get("format") != STATE_FORMAT_V2:
        raise ValueError("%s: unknown light-state format %s"
                         % (state_dir, state.get("format")))

    report = load_adapter_v2(model, state_dir / "adapter.pt",
                             map_location=map_location, allow_legacy=allow_legacy)

    inner = getattr(optimizer, "optimizer", optimizer)
    inner.load_state_dict(torch.load(state_dir / "optimizer.pt",
                                     map_location=map_location, weights_only=False))
    device = next(p.device for group in inner.param_groups for p in group["params"])
    L._optimizer_state_to_device(optimizer, device)

    scheduler_state = torch.load(state_dir / "scheduler.pt",
                                 map_location=map_location, weights_only=False)
    if scheduler_state:
        scheduler.load_state_dict(scheduler_state)

    L.restore_rng(torch.load(state_dir / "rng.pt", map_location="cpu", weights_only=False))
    L._apply_sampler_state(sampler, state["sampler"])
    state["adapter_report"] = report
    return state
