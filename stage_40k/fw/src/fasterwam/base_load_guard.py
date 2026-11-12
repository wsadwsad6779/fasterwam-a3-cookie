"""Guard the frozen-base load: expand proprio 8->14 and prove the base landed.

Two failure modes this closes, both silent in the stock code:

1. `FastWAM.load_checkpoint` (models/wan22/fastwam.py) loads proprio_encoder
   with strict=True. The upstream libero base is [4096, 8]; a force-aware model
   is [4096, 14], so the stock path raises. This expands the checkpoint tensor
   into the wider parameter, copying the first 8 columns and zero-filling the
   rest, so at step 0 the proprio token is bit-identical to the 8-dim model's.

2. The same call loads `mot` with strict=False. LoRALinear subclasses nn.Linear
   and rebinds `self.weight = base.weight` (lora.py), so key names survive
   wrapping and all 1393 base tensors do load -- but if that ever stops being
   true, strict=False drops them without a word and training starts from random
   noise. After the load, every checkpoint key must exist in the model and
   compare equal.

Installed as a runtime patch; no FasterWAM source file is rewritten.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def _record(entry):
    # One ledger file is appended to by every process in a rehearsal, so an event
    # is only attributable once it carries the stage that wrote it.
    entry = dict(entry)
    entry.setdefault("stage", os.environ.get("FW_LEDGER_STAGE", "unlabelled"))
    entry.setdefault("pid", os.getpid())
    logger.warning("base_load_guard: %s", json.dumps(entry, sort_keys=True))
    path = os.environ.get("FW_BASE_LOAD_LEDGER", "")
    if path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, sort_keys=True) + "\n")


def install():
    from fasterwam.models.wan22.fastwam import FastWAM

    if getattr(FastWAM.load_checkpoint, "_base_guard", False):
        return False
    orig = FastWAM.load_checkpoint

    def load_checkpoint(self, path, optimizer=None):
        enc = self.proprio_encoder
        expansion = {"action": "no_proprio_encoder"}

        if enc is not None:
            target_in = int(enc.in_features)
            real_load = enc.load_state_dict
            seen = {}

            def expanding_load(state_dict, strict=True, assign=False):
                if set(state_dict) != {"weight", "bias"}:
                    raise RuntimeError(
                        "proprio_encoder checkpoint has unexpected keys %s; refusing to guess"
                        % sorted(state_dict))
                w = state_dict["weight"]
                b = state_dict["bias"]
                src_in = int(w.shape[1])
                if src_in > target_in:
                    raise RuntimeError(
                        "%s: checkpoint proprio_encoder is %d wide but the model is %d; "
                        "truncating trained force columns would be silent data loss."
                        % (path, src_in, target_in))
                if src_in == target_in:
                    seen.update(action="exact", src_in=src_in, target_in=target_in)
                    return real_load(state_dict, strict=strict, assign=assign)
                new_w = torch.zeros_like(enc.weight)
                new_w[:, :src_in] = w.to(device=new_w.device, dtype=new_w.dtype)
                new_b = b.to(device=enc.bias.device, dtype=enc.bias.dtype)
                seen.update(action="expanded_zero_filled", src_in=src_in,
                            target_in=target_in, zero_columns=target_in - src_in,
                            src_dtype=str(w.dtype), src_absmax=float(w.abs().max()))
                return real_load({"weight": new_w, "bias": new_b},
                                 strict=strict, assign=assign)

            enc.load_state_dict = expanding_load
            try:
                payload = orig(self, path, optimizer)
            finally:
                del enc.load_state_dict
            expansion = dict(seen)
            if not expansion:
                raise RuntimeError(
                    "%s: proprio_encoder.load_state_dict was never called; the checkpoint "
                    "has no proprio_encoder weights and the model would keep random init."
                    % path)
            if expansion["action"] == "expanded_zero_filled":
                src_in = expansion["src_in"]
                src_w = payload["proprio_encoder"]["weight"].to(enc.weight.device, enc.weight.dtype)
                src_b = payload["proprio_encoder"]["bias"].to(enc.bias.device, enc.bias.dtype)
                with torch.no_grad():
                    if not bool((enc.weight[:, src_in:] == 0).all()):
                        raise RuntimeError("new proprio columns are not exactly zero")
                    if not torch.equal(enc.weight[:, :src_in], src_w):
                        raise RuntimeError("the original proprio columns did not survive expansion")
                    if not torch.equal(enc.bias, src_b):
                        raise RuntimeError("proprio bias did not survive expansion")
                expansion["zero_columns_verified"] = True
                expansion["copied_columns_verified"] = True
                expansion["bias_verified"] = True
            elif expansion["action"] == "exact":
                with torch.no_grad():
                    if not torch.equal(enc.weight, payload["proprio_encoder"]["weight"]
                                       .to(enc.weight.device, enc.weight.dtype)):
                        raise RuntimeError("proprio weight did not survive the exact load")
                expansion["weight_verified"] = True

        else:
            payload = orig(self, path, optimizer)

        _record({"event": "proprio_load", "path": str(path), **expansion})

        if "mot" not in payload:
            raise RuntimeError("%s: checkpoint has no `mot`; nothing to verify" % path)
        ckpt = payload["mot"]
        live = self.mot.state_dict()
        absent = sorted(k for k in ckpt if k not in live)
        if absent:
            raise RuntimeError(
                "%s: %d checkpoint tensors have no home in the model (strict=False would "
                "have dropped them silently), e.g. %s. The base did NOT load."
                % (path, len(absent), absent[:5]))
        shape_bad = [k for k in ckpt if tuple(live[k].shape) != tuple(ckpt[k].shape)]
        if shape_bad:
            raise RuntimeError("%s: mot shape mismatch on %s" % (path, shape_bad[:5]))

        verify_values = os.environ.get("FW_VERIFY_BASE_LOAD", "1") != "0"
        entry = {"event": "mot_load", "path": str(path), "checkpoint_tensors": len(ckpt),
                 "model_tensors": len(live), "absent": 0, "shape_mismatch": 0,
                 "value_verified": verify_values}
        if verify_values:
            bad = []
            elements = 0
            for k, v in ckpt.items():
                elements += int(v.numel())
                if not torch.equal(live[k].detach().to("cpu"), v.to("cpu")):
                    bad.append(k)
                    if len(bad) >= 5:
                        break
            if bad:
                raise RuntimeError(
                    "%s: %d mot tensors differ from the checkpoint after load, e.g. %s"
                    % (path, len(bad), bad))
            entry["verified_elements"] = elements
        _record(entry)
        return payload

    load_checkpoint._base_guard = True
    FastWAM.load_checkpoint = load_checkpoint
    return True