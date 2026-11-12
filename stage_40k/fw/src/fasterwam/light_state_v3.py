"""v3 light state: persist exactly the tensors the optimizer owns.

v1 keyed the save set on the name pattern ``lora_`` and silently dropped
``proprio_encoder``, which ``trainer.py`` puts in the AdamW list. A v2 draft keyed
on ``requires_grad`` instead; on the live model that is 1637 tensors /
8,710,770,916 elements, because the LoRA train-mode helper freezes by pattern
without clearing the flag on everything else -- writing it would have cost ~9 GB
per save. Optimizer membership is the only set that means "training changed
this", so v3 selects by parameter identity and refuses to write when the
projected size exceeds the caller's budget.

The optimizer / scheduler / RNG / sampler plumbing is reused from
``fasterwam.light_state`` unchanged; only the adapter selection and the state
format tag differ. ``save_light_state`` / ``load_light_state`` here are
signature-compatible with the v1 pair so a runtime patch can swap them in.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import torch

from fasterwam import light_state as v1

ADAPTER_FORMAT_V3 = "fasterwam-lora-adapter-v3"
STATE_FORMAT_V3 = "fasterwam-lora-light-state-v3"
V1_ADAPTER_FORMAT = "fasterwam-lora-adapter-v1"
V1_STATE_FORMAT = "fasterwam-lora-light-state-v1"

_WIDTH = {torch.bfloat16: 2, torch.float16: 2, torch.float32: 4, torch.float64: 8}


def _inner(optimizer):
    return getattr(optimizer, "optimizer", optimizer)


def optimizer_membership(optimizer):
    inner = _inner(optimizer)
    return [p for g in inner.param_groups for p in g["params"]]


def select_saved(model, optimizer):
    """Name -> tensor for every parameter the optimizer can modify.

    Identity first, because that is the truth. If the optimizer holds tensors
    that are not the model's own parameters (a sharding wrapper would), fall back
    to the trainer's construction order and accept it only when the shape/dtype
    sequence matches exactly.
    """
    named = list(model.named_parameters())
    members = optimizer_membership(optimizer)
    ids = {id(p): n for n, p in named}
    hits = [(n, p) for p in members if id(p) in ids for n in [ids[id(p)]]]
    method = "identity"
    if len(hits) != len(members):
        cand = [p for p in model.dit.parameters() if p.requires_grad]
        proprio = getattr(model, "proprio_encoder", None)
        if proprio is not None:
            cand += [p for p in proprio.parameters() if p.requires_grad]
        cand_names = []
        for n, p in named:
            if id(p) in {id(c) for c in cand}:
                cand_names.append((n, p))
        sig = [(tuple(p.shape), str(p.dtype)) for p in cand]
        msig = [(tuple(p.shape), str(p.dtype)) for p in members]
        if sig != msig:
            raise RuntimeError(
                "optimizer membership unresolved: identity matched %d of %d "
                "and the trainer-order fallback does not reproduce it "
                "(shapes %s vs %s). Refusing to guess what to save."
                % (len(hits), len(members), msig[:4], sig[:4])
            )
        hits, method = cand_names, "positional_verified"
    seen = set()
    out = []
    for n, p in hits:
        if n in seen:
            continue
        seen.add(n)
        out.append((n, p))
    return {"selected": out, "method": method, "optimizer_members": len(members),
            "selected_tensors": len(out)}


def member_name_order(model, optimizer):
    """Parameter names in flat ``param_groups`` order, duplicates kept.

    ``Optimizer.state_dict()`` keys its per-tensor state by that flat index, and
    ``load_state_dict`` writes the stored moments back by index alone. Two runs
    whose trainable sets differ -- a force module added, a freeze pattern
    reordered -- would therefore hand one tensor's Adam moments to another with
    no error at all. Recording the order makes that a mismatch we can refuse.
    Returns ``(None, reason)`` when the index cannot be resolved to names.
    """
    named = list(model.named_parameters())
    members = optimizer_membership(optimizer)
    ids = {}
    for n, p in named:
        ids.setdefault(id(p), n)
    if any(id(p) not in ids for p in members):
        return None, "optimizer holds tensors that are not model parameters"
    return [ids[id(p)] for p in members], None


def projected_bytes(selected):
    return sum(int(p.numel()) * _WIDTH.get(p.dtype, 4) for _, p in selected)


def free_bytes(path):
    d = Path(path)
    while not d.exists() and d != d.parent:
        d = d.parent
    return int(shutil.disk_usage(str(d)).free)


def coverage(model, optimizer):
    """Three different sets, kept apart on purpose."""
    named = list(model.named_parameters())
    sel = select_saved(model, optimizer)
    sel_names = {n for n, _ in sel["selected"]}
    req = [n for n, p in named if p.requires_grad]
    lora = [n for n in req if "lora_" in n]
    return {
        "named_tensors": len(named),
        "named_elements": sum(int(p.numel()) for _, p in named),
        "requires_grad_tensors": len(req),
        "requires_grad_elements": sum(int(p.numel()) for n, p in named if n in set(req)),
        "optimizer_members": sel["optimizer_members"],
        "saved_tensors": len(sel_names),
        "saved_elements": sum(int(p.numel()) for n, p in named if n in sel_names),
        "saved_lora_tensors": len([n for n in sel_names if "lora_" in n]),
        "saved_nonlora_names": sorted(n for n in sel_names if "lora_" not in n),
        "requires_grad_but_not_optimised": len([n for n in req if n not in sel_names]),
        "requires_grad_but_not_optimised_elements": sum(
            int(p.numel()) for n, p in named if p.requires_grad and n not in sel_names),
        "kept_by_v1": len(lora),
        "kept_by_v1_elements": sum(int(p.numel()) for n, p in named if n in set(lora)),
        "dropped_by_v1": sorted(n for n, p in named
                                if p.requires_grad and "lora_" not in n and n in sel_names),
        "dropped_by_v1_elements": sum(int(p.numel()) for n, p in named
                                      if p.requires_grad and "lora_" not in n and n in sel_names),
        "selection_method": sel["method"],
        "projected_adapter_bytes": projected_bytes(sel["selected"]),
    }


def save_adapter_v3(model, optimizer, path, meta=None, max_bytes=None):
    path = Path(path)
    cov = coverage(model, optimizer)
    sel = select_saved(model, optimizer)["selected"]
    est = projected_bytes(sel)
    if max_bytes is None:
        max_bytes = int(free_bytes(path.parent) * 0.25)
    if est > max_bytes:
        raise RuntimeError("%s: v3 adapter would write %d B but the budget is %d B; "
                           "nothing was written." % (path, est, max_bytes))
    payload = {"format": ADAPTER_FORMAT_V3,
               "lora": {n: p.detach().to("cpu", copy=True) for n, p in sel if "lora_" in n},
               "extra_trainable": {n: p.detach().to("cpu", copy=True) for n, p in sel
                                   if "lora_" not in n},
               "meta": dict(meta or {}, coverage=cov, projected_bytes=est)}
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    actual = tmp.stat().st_size
    if actual > max_bytes:
        os.remove(tmp)
        raise RuntimeError("%s: wrote %d B, over budget %d B; removed." % (path, actual, max_bytes))
    os.replace(tmp, path)
    return {"path": str(path), "bytes": actual, "coverage": cov}


def load_adapter_v3(model, path, map_location="cpu", allow_legacy=False):
    path = Path(path)
    payload = torch.load(path, map_location=map_location, weights_only=False)
    fmt = payload.get("format")
    named = dict(model.named_parameters())
    if fmt == V1_ADAPTER_FORMAT:
        if not allow_legacy:
            raise RuntimeError(
                "%s: v1 adapter holds only lora_ keys (%d tensors). Any non-LoRA "
                "parameter the optimizer stepped is absent, so this file cannot "
                "restore the trained model. Pass allow_legacy=True to load the "
                "partial restore anyway." % (path, len(payload["lora"])))
        blob = dict(payload["lora"])
    elif fmt == ADAPTER_FORMAT_V3:
        blob = dict(payload["lora"])
        blob.update(payload.get("extra_trainable", {}))
    else:
        raise ValueError("%s: not a recognised adapter checkpoint (%s)" % (path, fmt))
    unknown = sorted(set(blob) - set(named))
    if unknown:
        raise RuntimeError("%s: %d keys the model does not have, e.g. %s"
                           % (path, len(unknown), unknown[:5]))
    bad = ["%s file%s model%s" % (n, list(blob[n].shape), list(named[n].shape))
           for n in blob if list(blob[n].shape) != list(named[n].shape)]
    if bad:
        raise RuntimeError("%s: shape mismatch: %s" % (path, bad[:5]))
    with torch.no_grad():
        for n, v in blob.items():
            p = named[n]
            p.copy_(v.to(device=p.device, dtype=p.dtype))
    return {"format": fmt, "restored_tensors": len(blob),
            "restored_elements": sum(int(v.numel()) for v in blob.values()),
            "meta": payload.get("meta", {})}


def force_calibration_fingerprint(model):
    """Hash the force buffers so a resume can prove the units did not change.

    ``force_mean``/``force_std`` are buffers, so the adapter (optimizer members)
    never carries them: on resume they are rebuilt from whatever
    ``force.norm_stats.file`` the config points at. If that JSON is regenerated
    with a different corpus, channels or version, training would continue under
    a different normalisation than the one that produced the weights, silently.
    Comparing the fingerprint turns that into a hard error.
    """
    conditioner = getattr(model, "force_conditioner", None)
    if conditioner is None:
        conditioner = getattr(getattr(model, "module", None), "force_conditioner", None)
    if conditioner is None:
        return None
    enc = conditioner["history_encoder"]
    if float(enc.stats_are_set.detach().float().item()) != 1.0:
        raise RuntimeError(
            "force calibration buffers were never set; refusing to write a light state "
            "whose resume fingerprint would be meaningless."
        )
    digest = {}
    for name in ("force_mean", "force_std"):
        tensor = getattr(enc, name).detach().to("cpu", torch.float32).contiguous()
        digest["%s_sha256" % name] = hashlib.sha256(tensor.numpy().tobytes()).hexdigest()
    digest.update({
        "force_dim": int(enc.force_dim),
        "hist_len": int(enc.hist_len),
        "scale": float(enc.scale),
    })
    return digest


def _check_force_calibration(model, saved, state_dir):
    live = force_calibration_fingerprint(model)
    if saved is None and live is None:
        return None
    if saved is None:
        raise RuntimeError(
            "%s: light state written without a force calibration fingerprint while this model "
            "has force conditioning; refusing to resume under unverified units." % state_dir)
    if live is None:
        raise RuntimeError(
            "%s: light state carries force calibration %s but this model has no force "
            "conditioner." % (state_dir, saved))
    if live != saved:
        diff = {k: (saved.get(k), live.get(k)) for k in set(saved) | set(live)
                if saved.get(k) != live.get(k)}
        raise RuntimeError(
            "%s: force calibration differs between the light state and the model built from "
            "the current config (saved, live): %s. The stored weights were trained under "
            "different force units than this run would use." % (state_dir, diff))
    return live


def _check_member_order(state, model, optimizer, state_dir, fmt):
    """Refuse a resume whose optimizer slots do not line up with the stored ones.

    ``Optimizer.load_state_dict`` maps moments to parameters by flat index only.
    A v3 state records the names behind that index, so a trainable set that
    changed since the save -- different freeze pattern, force module added or
    renamed -- fails here instead of silently training on the wrong moments.
    """
    if fmt != STATE_FORMAT_V3:
        return None
    stored = state.get("optimizer_member_order")
    if stored is None:
        raise RuntimeError(
            "%s: v3 light state written without `optimizer_member_order`, so the "
            "optimizer moments in optimizer.pt cannot be matched to parameters by "
            "name. Refusing to apply them by index alone." % state_dir)
    live, reason = member_name_order(model, optimizer)
    if live is None:
        raise RuntimeError(
            "%s: cannot resolve this optimizer's parameter order (%s); the stored "
            "moments would be assigned by index with no way to check them."
            % (state_dir, reason))
    if list(live) != list(stored):
        first = next(i for i, (a, b) in enumerate(zip(stored, live)) if a != b)
        raise RuntimeError(
            "%s: the trainable set differs from the one that produced this state, "
            "so optimizer.pt's Adam moments would land on the wrong parameters. "
            "First mismatch at index %d: saved %r, this run %r (saved %d members, "
            "this run %d). Restore the original config or start from the base "
            "checkpoint instead." % (state_dir, first, stored[first], live[first],
                                     len(stored), len(live)))
    return live


def save_light_state(state_dir, *, model, optimizer, scheduler, sampler,
                     global_step, epoch, batch_in_epoch, extra=None,
                     adapter_meta=None, max_bytes=None):
    """Write adapter(v3) + optimizer + scheduler + RNG + counters; return manifest.

    Signature-compatible with ``light_state.save_light_state`` so the r4 runtime
    patch can call it without knowing which version is installed.
    """
    state_dir = Path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    order, order_reason = member_name_order(model, optimizer)

    def _replace(tmp_name, obj):
        tmp = state_dir / (tmp_name + ".tmp")
        torch.save(obj, tmp)
        os.replace(tmp, state_dir / tmp_name)

    info = save_adapter_v3(
        model, optimizer, state_dir / "adapter.pt",
        meta={"global_step": int(global_step), **(adapter_meta or {})},
        max_bytes=max_bytes)
    _replace("optimizer.pt", v1._optimizer_cpu_state(optimizer))
    _replace("scheduler.pt", getattr(scheduler, "state_dict", lambda: {})())
    _replace("rng.pt", v1.capture_rng())

    state = {
        "format": STATE_FORMAT_V3,
        "global_step": int(global_step),
        "epoch": int(epoch),
        "batch_in_epoch": int(batch_in_epoch),
        "sampler": v1._sampler_state(sampler),
        "extra": dict(extra or {}),
        "files": {"adapter": "adapter.pt", "optimizer": "optimizer.pt",
                  "scheduler": "scheduler.pt", "rng": "rng.pt"},
        "coverage": info["coverage"],
        "adapter_bytes": info["bytes"],
        "optimizer_member_order": order,
        "optimizer_member_order_reason": order_reason,
        "force_calibration": force_calibration_fingerprint(model),
    }
    tmp = state_dir / "trainer_state.json.tmp"
    tmp.write_text(json.dumps(state, indent=2))
    os.replace(tmp, state_dir / "trainer_state.json")
    return state


def load_light_state(state_dir, *, model, optimizer, scheduler, sampler,
                     map_location="cpu", allow_legacy=False):
    """Restore everything ``save_light_state`` wrote; return the manifest."""
    state_dir = Path(state_dir)
    state_file = state_dir / "trainer_state.json"
    if not state_file.exists():
        raise FileNotFoundError(f"{state_file} missing; refusing a partial restore")
    state = json.loads(state_file.read_text())
    fmt = state.get("format")
    if fmt == V1_STATE_FORMAT and not allow_legacy:
        raise RuntimeError(
            "%s: v1 light state. Its adapter dropped every non-LoRA trainable "
            "tensor, so resuming from it cannot reproduce the trained model. "
            "Pass allow_legacy=True to accept the partial restore." % state_dir)
    if fmt not in (STATE_FORMAT_V3, V1_STATE_FORMAT):
        raise ValueError(f"{state_dir}: unknown light-state format {fmt}")

    load_adapter_v3(model, state_dir / "adapter.pt", map_location=map_location,
                    allow_legacy=allow_legacy)
    _check_force_calibration(model, state.get("force_calibration"), state_dir)

    inner = _inner(optimizer)
    _check_member_order(state, model, optimizer, state_dir, fmt)
    inner.load_state_dict(torch.load(state_dir / "optimizer.pt",
                                     map_location=map_location, weights_only=False))
    device = next(p.device for group in inner.param_groups for p in group["params"])
    v1._optimizer_state_to_device(optimizer, device)

    scheduler_state = torch.load(state_dir / "scheduler.pt",
                                 map_location=map_location, weights_only=False)
    if scheduler_state:
        scheduler.load_state_dict(scheduler_state)

    v1.restore_rng(torch.load(state_dir / "rng.pt", map_location="cpu", weights_only=False))
    v1._apply_sampler_state(sampler, state["sampler"])
    return state