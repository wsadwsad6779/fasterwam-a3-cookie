"""Publish a v3 light state as a merged full-model checkpoint.

`scripts/export_lora_merged.py` reads `payload["mot"]` out of the adapter file.
A v3 adapter has no `mot` key at all -- it holds LoRA deltas plus the non-LoRA
tensors the optimizer stepped, and deliberately no base weights. Publishing from
it therefore requires the frozen base first, loaded the same way the trainer
loads it at step 0 and through the same `base_load_guard`, so the 8->14 proprio
expansion and the elementwise "did the base actually land" check happen here too
instead of being assumed.

Order is load-bearing: base load, adapter overlay, merge. Overlaying before the
base load would let the base overwrite the trained tensors.

The merge is checked against an independently computed `W + (B@A)*scaling` on the
same device, bitwise, for all 392 wrapped matrices; every other tensor must be
untouched. The companion `verify_publish65.py` reloads the written file through
the deployment path.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

import fasterwam.base_load_guard as base_load_guard
from fasterwam.light_state_v3 import force_calibration_fingerprint, load_adapter_v3
from fasterwam.lora import LoRALinear, apply_lora, merge_lora
from fasterwam.runtime import _mixed_precision_to_model_dtype, _normalize_mixed_precision
from fasterwam.utils.config_resolvers import register_default_resolvers

register_default_resolvers()

FORCE_MARKERS = ("force_conditioner.", "force_mean", "force_std", "stats_are_set")


def _is_force_key(key: str) -> bool:
    return any(marker in key for marker in FORCE_MARKERS)


def _cpu(t) -> torch.Tensor:
    return t.detach().to("cpu").clone()


def _equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and a.dtype == b.dtype and bool(torch.equal(a, b))


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    base_ckpt = cfg.get("base_ckpt")
    light_dir = cfg.get("light_dir")
    output = cfg.get("output")
    record_path = cfg.get("record")
    missing_args = [n for n, v in (("base_ckpt", base_ckpt), ("light_dir", light_dir),
                                   ("output", output)) if v in (None, "None")]
    if missing_args:
        raise RuntimeError("this script needs %s on the command line (use +%s=...)"
                           % (",".join(missing_args), "+".join(missing_args)))
    light_dir = Path(str(light_dir))
    out_path = Path(str(output))
    state_file = light_dir / "trainer_state.json"
    if not state_file.exists():
        raise RuntimeError(f"{state_file} missing; refusing to publish from a partial light state")

    record: dict = {"event": "publish_merged", "ts": time.time(),
                    "base_ckpt": str(base_ckpt), "light_dir": str(light_dir),
                    "output": str(out_path)}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = _mixed_precision_to_model_dtype(_normalize_mixed_precision(cfg.mixed_precision))
    record["device"] = str(device)
    record["model_dtype"] = str(dtype)

    model = instantiate(cfg.model, model_dtype=dtype, device=device)
    n_layers, n_trainable = apply_lora(model, cfg.lora)
    record["lora_layers"] = int(n_layers)
    record["lora_trainable_params"] = int(n_trainable)
    if n_layers == 0:
        raise RuntimeError("LoRA wrapped nothing; the published file would be the base verbatim")

    record["base_load_guard_installed_here"] = bool(base_load_guard.install())
    base_payload = model.load_checkpoint(str(base_ckpt), optimizer=None)
    record["base_mot_tensors"] = len(base_payload["mot"])

    adapter = load_adapter_v3(model, light_dir / "adapter.pt", map_location=device)
    record["adapter_restored_tensors"] = int(adapter["restored_tensors"])
    record["adapter_restored_elements"] = int(adapter["restored_elements"])

    live = dict(model.mot.state_dict())
    wrapped = [name for name, mod in model.mot.named_modules() if isinstance(mod, LoRALinear)]
    refs, deltas = {}, {}
    for name in wrapped:
        mod = model.mot.get_submodule(name)
        refs[f"{name}.weight"] = mod.weight.detach().clone()
        deltas[f"{name}.weight"] = (mod.lora_B.detach() @ mod.lora_A.detach()) * mod.scaling
    force_keys_before = sorted(k for k in live if _is_force_key(k))
    force_before = {k: _cpu(live[k]) for k in force_keys_before}
    untouched = {
        k: _cpu(live[k]) for k in live
        if k not in refs and not k.endswith(".lora_A") and not k.endswith(".lora_B")
        and not _is_force_key(k)
    }
    proprio_before = (_cpu(model.proprio_encoder.weight), _cpu(model.proprio_encoder.bias))
    record["wrapped_matrices"] = len(refs)
    record["untouched_mot_tensors_verified"] = len(untouched)
    n_wrapped = len(refs)

    merged = merge_lora(model)
    if merged != len(wrapped):
        raise RuntimeError(f"merge_lora folded {merged} layers but {len(wrapped)} were applied")

    after = dict(model.mot.state_dict())
    stray_lora = sorted(k for k in after if "lora_" in k)
    if stray_lora:
        raise RuntimeError(f"merged state_dict still carries LoRA tensors: {stray_lora[:5]}")

    merge_bad = []
    for k, ref in refs.items():
        got = after[k]
        want = ref + deltas[k]
        if not _equal(got, want):
            merge_bad.append({"key": k, "max_abs_err": float((got.float() - want.float()).abs().max()),
                              "bitwise": False})
        del want
    if merge_bad:
        raise RuntimeError("merge is not base+(B@A)*scaling bitwise: %s" % merge_bad[:5])
    for k, want in untouched.items():
        if not _equal(_cpu(after[k]), want):
            raise RuntimeError(f"non-LoRA tensor {k} changed during merge")
    force_keys_after = sorted(k for k in after if _is_force_key(k))
    if force_keys_after != force_keys_before:
        raise RuntimeError("force key set changed during merge: %s vs %s"
                           % (force_keys_after[:5], force_keys_before[:5]))
    for k in force_keys_after:
        if not _equal(_cpu(after[k]), force_before[k]):
            raise RuntimeError(f"force tensor {k} changed during merge")
    del refs, deltas

    state = json.loads(state_file.read_text())
    declared_fp = force_calibration_fingerprint(model)
    if declared_fp is not None and state.get("force_calibration") is None:
        raise RuntimeError(
            "%s carries no force calibration fingerprint, so nothing proves the units "
            "this model normalises with are the units the stored force weights were "
            "trained under. Refusing to publish an unverifiable file." % state_file)
    if state.get("force_calibration") is not None and declared_fp != state["force_calibration"]:
        raise RuntimeError(
            "%s: the run trained under force units %s but this model's calibration is %s; "
            "refusing to publish a file that would declare the wrong units."
            % (state_file, state["force_calibration"], declared_fp))
    payload = {
        "mot": after,
        "proprio_encoder": {"weight": _cpu(model.proprio_encoder.weight),
                            "bias": _cpu(model.proprio_encoder.bias)},
        "step": int(state["global_step"]),
        "torch_dtype": str(dtype),
        "force_calibration": declared_fp,
    }
    for key, want in zip(("weight", "bias"), proprio_before):
        if not _equal(payload["proprio_encoder"][key], want):
            raise RuntimeError(f"proprio_encoder.{key} did not survive into the published payload")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".tmp")
    torch.save(payload, tmp)
    if tmp.stat().st_size < 1 << 20:
        raise RuntimeError(f"{tmp} is {tmp.stat().st_size} B; a full merged checkpoint cannot be that small")
    tmp.replace(out_path)

    buffers = [k for k in force_keys_after
               if any(m in k for m in ("force_mean", "force_std", "stats_are_set"))]
    record.update(
        merged_layers=merged,
        mot_tensors=len(after),
        force_tensors=len(force_keys_after),
        force_elements=sum(int(after[k].numel()) for k in force_keys_after),
        force_keys=force_keys_after,
        force_buffer_keys=buffers,
        published_step=payload["step"],
        published_force_calibration=declared_fp,
        output_bytes=out_path.stat().st_size,
        merge_check="all %d wrapped matrices equal base+(B@A)*scaling bitwise" % n_wrapped,
    )
    if record_path not in (None, "None"):
        rp = Path(str(record_path))
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(json.dumps(record, indent=2, sort_keys=True))
    print(json.dumps({k: v for k, v in record.items() if k != "force_keys"}, indent=2, sort_keys=True))
    print("published -> %s (%d B)" % (out_path, out_path.stat().st_size))


if __name__ == "__main__":
    main()
