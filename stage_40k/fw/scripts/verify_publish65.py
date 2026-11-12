"""Task #65 stage 5: reload a published merged checkpoint the way deployment does.

The eight checkpoints that go to Hugging Face are full merged models, loaded by
`FastWAM.load_checkpoint` with `strict=False`. `strict=False` is exactly how
r4/r5 lost `proprio_encoder`: a key the file does not carry is dropped without an
error. This script therefore does not trust the return value -- it compares the
live model against the file tensor by tensor, and then proves that the guard
which is supposed to catch that class of failure actually fires.

The controls each carry every force tensor plus one base tensor, so a raise can
only come from the specific defect the control is named for. They test the
guarantees `base_load_guard` and `FastWAM._force_calibration_audit` actually make:
a key the model has no home for, and a force buffer whose value no longer matches
the units the file declares. They deliberately do not test "a wrong value in a
base tensor raises" -- the guard compares the model against the file, so a file
that is internally consistent loads successfully by construction. Detecting that
a file disagrees with an independent source is P2 and P3a, which compare against
the light state. P6d records that limit instead of pretending it away.

Checks
  P0/P0b layout            -- mot + proprio_encoder + step, no LoRA leftovers
  P1 key set identity      -- the file covers the plain model exactly
  P2 elementwise landing   -- every tensor in the file equals the live tensor
  P3a force params         -- came from the light state, bitwise, after mapping
                             action_expert.* onto the mot namespace
  P3b force buffers        -- equal the calibration JSON the config names, as that
                             JSON is represented in the model dtype, bitwise
  P4 fingerprint           -- hashes to the fingerprint recorded at save time
  P5 proprio               -- 14-wide proprio survived the whole chain
  P6 hard-fail controls    -- a renamed key and a mis-calibrated force buffer raise
  P6b positive control     -- a well-formed subset checkpoint still loads
  P6c calibration audit    -- an undeclared or mismatched fingerprint raises
  P6d stated limit         -- a perturbed base value loads; the guard verifies
                             fidelity, not ground truth
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

import fasterwam.base_load_guard as base_load_guard
from fasterwam.light_state_v3 import force_calibration_fingerprint
from fasterwam.runtime import _mixed_precision_to_model_dtype, _normalize_mixed_precision
from fasterwam.utils.config_resolvers import register_default_resolvers

register_default_resolvers()

FORCE_MARKERS = ("force_conditioner.", "force_mean", "force_std", "stats_are_set")
BUFFER_LEAVES = ("force_mean", "force_std", "stats_are_set")
# The trainer addresses the action expert as `action_expert.*`; the published
# `mot` state_dict addresses the same tensors as `mixtures.action.*`. Comparing
# the two domains without mapping them reports every trained tensor as absent,
# which is exactly what the first run of this script did.
MODEL_HEAD = "action_expert."
MOT_HEAD = "mixtures.action."


def _to_mot(name: str) -> str:
    if name.startswith(MOT_HEAD):
        return name
    if name.startswith(MODEL_HEAD):
        return MOT_HEAD + name[len(MODEL_HEAD):]
    raise RuntimeError("%s: in neither the action_expert nor the mixtures.action namespace"
                       % name)


def _norm_stats_files(obj, out):
    """Every `norm_stats.file` the composed config declares, wherever it sits."""
    if isinstance(obj, dict):
        ns = obj.get("norm_stats")
        if isinstance(ns, dict) and "file" in ns:
            out.append(str(ns["file"]))
        for v in obj.values():
            _norm_stats_files(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _norm_stats_files(v, out)
    return out


def _is_force_key(k: str) -> bool:
    return any(m in k for m in FORCE_MARKERS)


def _is_buffer_key(k: str) -> bool:
    return any(k.endswith(n) for n in BUFFER_LEAVES)


class _Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record.getMessage())


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    merged_path = Path(str(cfg["merged"]))
    light_dir = Path(str(cfg["light_dir"]))
    calib_path = Path(str(cfg["calib"]))
    out_path = Path(str(cfg["report"]))
    adapter_path = light_dir / "adapter.pt"
    expected_fp = json.loads((light_dir / "trainer_state.json").read_text()).get("force_calibration")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = _mixed_precision_to_model_dtype(_normalize_mixed_precision(cfg.mixed_precision))
    checks = []

    def check(name, ok, evidence):
        checks.append({"name": name, "ok": bool(ok), "evidence": evidence})
        print(("PASS " if ok else "FAIL ") + name + " :: "
              + json.dumps(evidence, sort_keys=True, default=str)[:900])

    payload = torch.load(merged_path, map_location="cpu", weights_only=False)
    published = payload["mot"]
    adapter = torch.load(adapter_path, map_location="cpu", weights_only=False)
    trained = dict(adapter["lora"])
    trained.update(adapter.get("extra_trainable", {}))
    calib = json.loads(calib_path.read_text())
    force_keys = sorted(k for k in published if _is_force_key(k))
    param_keys = [k for k in force_keys if not _is_buffer_key(k)]
    buffer_keys = [k for k in force_keys if _is_buffer_key(k)]
    trained_force = sorted(k for k in trained if _is_force_key(k))

    check("P0 published file layout",
          set(payload) >= {"mot", "proprio_encoder", "step", "force_calibration"}
          and not sorted(k for k in published if "lora_" in k),
          {"keys": sorted(payload), "step": payload.get("step"),
           "torch_dtype": payload.get("torch_dtype"), "mot_tensors": len(published),
           "bytes": merged_path.stat().st_size,
           "force_calibration": payload.get("force_calibration"),
           "lora_leftovers": sorted(k for k in published if "lora_" in k)[:5]})

    model = instantiate(cfg.model, model_dtype=dtype, device=device)
    cap = _Capture()
    logging.getLogger("fasterwam.models.wan22.fastwam").addHandler(cap)

    live = model.mot.state_dict()
    file_only = sorted(k for k in published if k not in live)
    model_only = sorted(k for k in live if k not in published)
    check("P1 key set identity with a plain force-conditioned model",
          not file_only and not model_only,
          {"file_only": file_only[:5], "model_only": model_only[:5],
           "file_tensors": len(published), "model_tensors": len(live),
           "model_force_keys": len(sorted(k for k in live if _is_force_key(k)))})

    base_load_guard.install()
    returned = model.load_checkpoint(str(merged_path), optimizer=None)
    live = model.mot.state_dict()
    bad = []
    for k, v in published.items():
        ref = live[k].detach().to("cpu")
        if ref.shape != v.shape or ref.dtype != v.dtype or not torch.equal(ref, v):
            bad.append(k)
            if len(bad) >= 5:
                break
    check("P2 every published tensor landed in the live model", not bad,
          {"mismatched": bad, "compared": len(published),
           "returned_mot_tensors": len(returned["mot"]),
           "force_status_log": [m for m in cap.records if "force" in m.lower()][:3]})

    mapped, collisions = {}, []
    for k in trained_force:
        m = _to_mot(k)
        if m in mapped:
            collisions.append((k, mapped[m], m))
        mapped[m] = k
    bad = [(k, "no light-state tensor maps to it") for k in param_keys if k not in mapped]
    for k in [k for k in param_keys if k in mapped]:
        want, got = trained[mapped[k]].to("cpu"), live[k].detach().to("cpu")
        if want.shape != got.shape or want.dtype != got.dtype or not torch.equal(want, got):
            bad.append((k, "differs from the light state"))
    check("P3a trained force tensors came from the light state, bitwise",
          not bad and not collisions and sorted(mapped) == sorted(param_keys)
          and not any("lora_" in k for k in mapped),
          {"force_param_tensors": len(param_keys), "bad": bad[:5],
           "collisions": collisions[:3],
           "light_state_force_param_tensors": len(trained_force),
           "mapped_into_mot_namespace": len(mapped),
           "name_mapping": {"from": MODEL_HEAD, "to": MOT_HEAD},
           "unmapped_published_force_params": sorted(set(param_keys) - set(mapped))[:5],
           "unmapped_light_state_force": sorted(set(mapped) - set(param_keys))[:5],
           "force_elements": sum(int(published[k].numel()) for k in param_keys)})

    enc = model.force_conditioner["history_encoder"]
    mean_buf = enc.force_mean.detach().to("cpu")
    std_buf = enc.force_std.detach().to("cpu")
    rounded_mean = torch.tensor(calib["mean"], dtype=torch.float32).to(mean_buf.dtype).to(torch.float32)
    rounded_std = torch.tensor(calib["std"], dtype=torch.float32).to(std_buf.dtype).to(torch.float32)
    declared_calib = sorted(set(_norm_stats_files(
        OmegaConf.to_container(cfg, resolve=True), [])))
    check("P3b calibration buffers equal the frozen JSON, in the model dtype, bitwise",
          bool(torch.equal(mean_buf.to(torch.float32), rounded_mean))
          and bool(torch.equal(std_buf.to(torch.float32), rounded_std))
          and float(enc.stats_are_set) == 1.0
          and list(calib["channels"]) == [0, 1, 2, 12, 13, 16]
          and declared_calib == [str(calib_path)],
          {"mean_bitwise_equal": bool(torch.equal(mean_buf.to(torch.float32), rounded_mean)),
           "std_bitwise_equal": bool(torch.equal(std_buf.to(torch.float32), rounded_std)),
           "buffer_dtype": str(mean_buf.dtype),
           "float32_json_vs_buffer_absmax_err":
               float((mean_buf.to(torch.float32)
                      - torch.tensor(calib["mean"], dtype=torch.float32)).abs().max()),
           "stats_are_set": float(enc.stats_are_set), "buffer_keys": buffer_keys,
           "calib_channels": list(calib["channels"]),
           "norm_stats_files_in_config": declared_calib})

    fp_live = force_calibration_fingerprint(model)
    check("P4 published buffers, file declaration and light state all give one fingerprint",
          expected_fp is not None and fp_live == expected_fp
          and payload.get("force_calibration") == expected_fp,
          {"from_light_state": expected_fp, "from_published_model": fp_live,
           "declared_in_file": payload.get("force_calibration")})

    pw = model.proprio_encoder.weight.detach().to("cpu")
    pa = trained.get("proprio_encoder.weight")
    same = pa is not None and torch.equal(pw, pa.to(pw.dtype))
    check("P5 14-wide proprio survived base+overlay+merge+reload",
          pw.shape[1] == 14 and same,
          {"shape": list(pw.shape), "equals_light_state": bool(same),
           "force_column_absmax": float(pw[:, 8:].abs().max()),
           "note": "12 steps inside warmup at 5e-8 cannot move a bf16 4096-wide row; "
                   "the claim here is survival, not movement"})

    # ---- P6 -----------------------------------------------------------------
    scratch = merged_path.parent / "_guard_controls"
    scratch.mkdir(parents=True, exist_ok=True)
    base_key = sorted((k for k in published if not _is_force_key(k) and published[k].numel() > 1),
                      key=lambda k: int(published[k].numel()))[0]
    tiny = {"weight": pw.clone(), "bias": model.proprio_encoder.bias.detach().to("cpu").clone()}
    core = {k: published[k] for k in force_keys}
    mean_key = [k for k in buffer_keys if k.endswith("force_mean")]
    assert len(mean_key) == 1, buffer_keys
    mean_key = mean_key[0]
    good_fp = payload["force_calibration"]
    plain = dict(core, **{base_key: published[base_key]})
    cases = {
        "clean_subset": (plain, good_fp),
        "renamed_key": (dict(core, **{base_key + "_RENAMED": published[base_key]}), good_fp),
        # The file keeps the honest fingerprint but its force_mean no longer matches
        # it, so only the calibration audit can catch this one.
        "perturbed_force_buffer": (dict(core, **{
            mean_key: published[mean_key] + 1.0,
            base_key: published[base_key]}), good_fp),
        "no_fingerprint": (plain, None),
        "wrong_fingerprint": (plain, dict(good_fp, force_std_sha256="0" * 64)),
        # The guard compares the model against the file, so this one must load.
        "perturbed_base_value": (dict(core, **{base_key: published[base_key] + 1.0}), good_fp),
    }
    restore_keys = sorted(set(force_keys) | {base_key})

    def _restore_live():
        """Put the model back exactly as P2 found it before the next control."""
        with torch.no_grad():
            model.mot.load_state_dict({k: published[k] for k in restore_keys}, strict=False)
        back = model.mot.state_dict()
        drifted = [k for k in restore_keys
                   if not torch.equal(back[k].detach().to("cpu"), published[k])]
        if drifted:
            raise RuntimeError("live model could not be restored to the published file "
                               "between controls: %s" % drifted[:5])

    results = {}
    for label, (mot_blob, fp) in cases.items():
        p = scratch / ("%s.pt" % label)
        saved = {"mot": mot_blob, "proprio_encoder": tiny, "step": 1}
        if fp is not None:
            saved["force_calibration"] = fp
        torch.save(saved, p)
        try:
            model.load_checkpoint(str(p), optimizer=None)
        except Exception as exc:  # noqa: BLE001 -- raising is the expected outcome
            results[label] = {"raised": type(exc).__name__, "msg": str(exc)[:240]}
        else:
            results[label] = {"raised": None, "msg": "load succeeded"}
        _restore_live()
        results[label]["restored"] = True

    hard = {k: results[k] for k in ("renamed_key", "perturbed_force_buffer")}
    check("P6 base_load_guard and the calibration audit hard-fail on a bad file",
          bool(results["renamed_key"]["raised"])
          and bool(results["perturbed_force_buffer"]["raised"])
          and "no home" in results["renamed_key"]["msg"]
          and ("unit" in results["perturbed_force_buffer"]["msg"].lower()
               or "calibration" in results["perturbed_force_buffer"]["msg"].lower()),
          dict(hard, base_key=base_key, mean_key=mean_key))
    check("P6b a well-formed subset checkpoint loads (the guards are not always-on)",
          results["clean_subset"]["raised"] is None, results["clean_subset"])
    check("P6c the calibration audit refuses an undeclared or mismatched fingerprint",
          bool(results["no_fingerprint"]["raised"])
          and bool(results["wrong_fingerprint"]["raised"])
          and "calibration" in (results["no_fingerprint"]["msg"]
                                + results["wrong_fingerprint"]["msg"]).lower(),
          {k: results[k] for k in ("no_fingerprint", "wrong_fingerprint")})
    check("P6d stated limit: a wrong base value loads faithfully, so P2/P3a must catch it",
          results["perturbed_base_value"]["raised"] is None,
          dict(results["perturbed_base_value"],
               why="base_load_guard verifies file->model fidelity, not file vs an "
                   "independent source; the independent comparison is P2 (file vs live) "
                   "and P3a (published force params vs the light state)"))

    report = {"passed": sum(1 for c in checks if c["ok"]),
              "failed": [c["name"] for c in checks if not c["ok"]],
              "checks": checks,
              "merged_bytes": merged_path.stat().st_size,
              "base_bytes": Path(str(cfg["base_ckpt"])).stat().st_size}
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str))
    print(json.dumps({"passed": report["passed"], "failed": report["failed"]}, indent=2))
    return 0 if not report["failed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
