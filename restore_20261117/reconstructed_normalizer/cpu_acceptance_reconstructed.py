"""Req3 CPU acceptance of the RECONSTRUCTED 14D normalizer contract.

    NORMALIZER_IDENTITY=RECONSTRUCTED_DETERMINISTIC_NOT_ORIGINAL
    original_stats_sha256=UNKNOWN

Runs on the pod CPU (/opt/venv/bin/python): no GPU, no simulator, no merged
weights. Verifies the reconstructed stats against the ORIGINAL a3_fw 8D stats
and the ORIGINAL force calibration, exercises the real normalizer/processor
classes and eval40k's own build_proprio14/check_stats_width, and writes
RECONSTRUCTED_NORMALIZER_ACCEPTANCE.json. The synthetic observation used in the
P1/P2 check is a labeled CPU fixture, NOT a GPU result. This is diagnosis under
traceable reconstructed statistics, not proof of original byte identity.

Pod run:
    /opt/venv/bin/python cpu_acceptance_reconstructed.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

BASE = Path(os.environ.get("EVAL_RESTORE_BASE", "/workspace/eval_restore_20261117"))
FW = BASE / "deploy/stage_40k/fw"
EVAL40K_DIR = BASE / "deploy/stage_40k/dev/eval40k"
RECON_DIR = BASE / "reconstructed_normalizer"
RECONSTRUCTED = RECON_DIR / "dataset_stats_14d_RECONSTRUCTED.json"
ORIGINAL_8D = BASE / "archive_evidence/data/a3_fw/dataset_stats.json"
CALIB_PATH = BASE / "step_040000/force_calib_v1.json"
RUN_CFG = BASE / "step_040000/run_config.yaml"
FORCE_ARRAYS = BASE / "archive_evidence/data/a3_fw_force/arrays.npz"
OUT_DEFAULT = RECON_DIR / "RECONSTRUCTED_NORMALIZER_ACCEPTANCE.json"

NORMALIZER_IDENTITY = "RECONSTRUCTED_DETERMINISTIC_NOT_ORIGINAL"
ORIGINAL_STATS_SHA256 = "UNKNOWN"
RECONSTRUCTED_SHA256 = "47fbe4c4e3a40eeb4f76c2a06857914351dc863da36d8b8d4d4794b1d1b56d6b"
ORIGINAL_8D_SHA256 = "ed833168725b0b63179a8aa3f870543f119b6891305fb94d17b3357989ccfffd"
FORCE_CALIB_SHA256 = "2e821a8a019f966d0d2418df7f587d9d469ccd5596abcc39e9f20f5f830d1bd5"
STAT_KINDS = ("global_min", "global_max", "global_mean", "global_std", "global_q01", "global_q99")

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

for _p in reversed([FW / "src", EVAL40K_DIR]):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import eval40k  # noqa: E402

# Same import-order guarantee as the GPU driver: pin the force-aware fasterwam
# before anything else can bind the released action-only copy.
FW_IMPORT_FACTS = eval40k.use_stage_fw()

import force_stream  # noqa: E402
from fasterwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor  # noqa: E402
from fasterwam.datasets.lerobot.utils.normalizer import (  # noqa: E402
    LinearNormalizer, load_dataset_stats_from_json)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_cfg():
    """Load the RESOLVED original training config (run_config.yaml) read-only and redirect its
    dead historical force-calib / stats absolute paths to the restored calib and the reconstructed
    stats via marked mappings. run_config.yaml IS the training-time processor config
    (proprio_output_dim=14, action_output_dim=7, norm_default_mode=min/max), so this verifies
    'Processor uses training normalization' more faithfully than a fresh hydra compose of
    sim_libero, whose ${hydra:...} interpolations require a live HydraConfig that bare compose()
    never sets. The archived run_config.yaml on disk is never modified."""
    from omegaconf import OmegaConf

    import fasterwam.utils.config_resolvers as resolvers
    resolvers.register_default_resolvers()

    cfg = OmegaConf.load(str(RUN_CFG))
    OmegaConf.set_struct(cfg, False)
    if "force" in cfg.model and "norm_stats" in cfg.model.force:
        cfg.model.force.norm_stats.file = str(CALIB_PATH)
    cfg.data.train.pretrained_norm_stats = str(RECONSTRUCTED)
    OmegaConf.set_struct(cfg, True)
    OmegaConf.resolve(cfg)
    mappings = {"config_source": str(RUN_CFG),
                "config_source_sha256": sha256_file(RUN_CFG),
                "model.force.norm_stats.file": str(CALIB_PATH),
                "data.train.pretrained_norm_stats": str(RECONSTRUCTED)}
    return cfg, mappings


def make_normalizer(stats):
    shape_meta = {
        "action": [{"key": "default", "raw_shape": 7, "shape": 7}],
        "state": [{"key": "default", "raw_shape": 14, "shape": 14}],
    }
    return LinearNormalizer(shape_meta, False, "min/max", None, stats)


def fixture_proprio14(calib):
    # CPU fixture only: known base8 columns plus arange(18) raw force, so the
    # tare arithmetic can be checked bit-for-bit without the simulator.
    base8 = np.array([0.1, 0.2, 0.8, 0.0, 0.0, 0.0, 0.02, -0.02], dtype=np.float32)
    obs = {
        "observation.force": np.arange(18, dtype=np.float32),
        "observation.state": np.zeros(16, dtype=np.float32),
        "observation.eef_pose": np.array([0.1, 0.2, 0.8, 1.0, 0.0, 0.0, 0.0,
                                          0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
    }
    stub_w = SimpleNamespace(build_proprio=lambda o: base8.copy())
    p14 = eval40k.build_proprio14(obs, stub_w, calib, where="cpu_fixture")
    raw6 = force_stream.select_channels(obs[force_stream.FORCE_OBS_KEY], where="cpu_fixture")
    zero = np.asarray(calib["zero_load_ref_median"], dtype=np.float64)
    resid = (raw6.astype(np.float64) - zero).astype(np.float32)
    return obs, base8, p14, raw6, resid


def check_proprio_first8(recon_raw, orig_raw, stats, orig_stats):
    diffs = []
    for kind in STAT_KINDS:
        r = recon_raw["state"]["default"][kind][:8]
        o = orig_raw["state"]["default"][kind]
        assert r == o, f"state {kind}[0:8] differs from original a3_fw"
        rt = stats["state"]["default"][kind][:8]
        ot = orig_stats["state"]["default"][kind]
        assert torch.equal(rt, ot), f"state {kind}[0:8] tensors not bit-identical"
        diffs.append(float((rt.double() - ot.double()).abs().max()))
    detail = {"kinds_checked": list(STAT_KINDS), "width_first_block": 8,
              "identity": "exact JSON value equality AND float32 bit equality"}
    return detail, max(diffs)


def check_force_band(stats, calib, stats_path):
    std64 = np.asarray(calib["std"], dtype=np.float64)
    std_source = ("force_calib_v1.json std (global-zeroed residual std; equals the "
                  "build-algorithm recomputation to 1.11e-16 per RECONSTRUCTION_EVIDENCE)")
    if FORCE_ARRAYS.is_file():
        with np.load(str(FORCE_ARRAYS), allow_pickle=False) as arrays:
            if "force6" in arrays.files:
                force6 = np.asarray(arrays["force6"], dtype=np.float64)
            elif "force18" in arrays.files:
                force6 = np.asarray(arrays["force18"], dtype=np.float64)[
                    :, list(force_stream.CHANNELS)]
            else:
                force6 = None
        if force6 is not None:
            zeroed = force6 - np.asarray(calib["zero_load_ref_median"], dtype=np.float64)
            recomputed = zeroed.std(axis=0)
            assert np.max(np.abs(recomputed - std64)) <= 1e-9, \
                "recomputed residual std disagrees with force_calib std"
            std64 = recomputed
            std_source = "recomputed from archive force arrays (ddof=0), cross-checked vs calib"
    # The builder's float32 trick: band edges are +/- float32(round) of the
    # float64 residual std, so min/max mode becomes residual / std32 exactly.
    std32 = std64.astype(np.float32).astype(np.float64)
    entry = stats["state"]["default"]
    gmin = entry["global_min"].double().numpy()
    gmax = entry["global_max"].double().numpy()
    gstd = entry["global_std"].double().numpy()
    assert gmin.shape == (14,) and gmax.shape == (14,), "state band not 14-wide"
    assert np.array_equal(gmax[8:], std32), "force global_max != +std32"
    assert np.array_equal(gmin[8:], -std32), "force global_min != -std32"
    std_diff = float(np.max(np.abs(gstd[8:] - std64)))
    assert std_diff <= 1e-6, f"force global_std deviates from residual std by {std_diff}"
    gate = eval40k.check_stats_width(stats, str(stats_path), calib_std=calib["std"])
    band = gate["bands"]["default"]
    assert band["force_band_is_symmetric_std"], gate
    assert band["force_halfwidth_equals_calib_std"], gate
    detail = {"std_source": std_source,
              "std32": std32.tolist(),
              "global_min_force": gmin[8:].tolist(),
              "global_max_force": gmax[8:].tolist(),
              "band_edges_exact_float32": True,
              "check_stats_width": "passed (14-wide, half-width == calib std, atol 1e-6)"}
    return detail, std_diff


def check_action(recon_raw, orig_raw, stats, orig_stats):
    diffs = []
    for kind in STAT_KINDS:
        r = recon_raw["action"]["default"][kind]
        o = orig_raw["action"]["default"][kind]
        assert r == o, f"action {kind} differs from original a3_fw"
        rt = stats["action"]["default"][kind]
        ot = orig_stats["action"]["default"][kind]
        assert torch.equal(rt, ot), f"action {kind} tensors not bit-identical"
        diffs.append(float((rt.double() - ot.double()).abs().max()))
    detail = {"kinds_checked": list(STAT_KINDS), "action_dim": 7,
              "identity": "exact JSON value equality AND float32 bit equality"}
    return detail, max(diffs)


def check_p1_p2(calib, stats):
    obs, base8, p14, raw6, resid = fixture_proprio14(calib)
    expected_raw = np.arange(18, dtype=np.float32)[list(force_stream.CHANNELS)]
    # P2 (force_history) stays RAW newtons: select_channels must not tare.
    assert np.array_equal(raw6, expected_raw), "select_channels tared or altered raw force"
    assert force_stream.SCALE == 1.0, force_stream.SCALE
    stream = force_stream.ForceStream(hist_len=force_stream.HIST_LEN)
    stream.push(obs[force_stream.FORCE_OBS_KEY])
    history, mask = stream.snapshot()
    assert np.array_equal(history[-1], raw6), "ForceStream history is not raw newtons"
    assert not mask[:-1].any() and mask[-1], "left-padding mask wrong"
    # P1 (proprio cols 8-13) is the zero-load-tared residual, float64 then float32.
    assert np.array_equal(p14[:8], base8), "base8 columns altered"
    assert np.array_equal(p14[8:], resid), "proprio force cols != raw6 - zero_load_ref_median"
    assert not np.array_equal(raw6, resid), "tare was a no-op; fixture cannot prove P1"
    std32 = np.asarray(calib["std"], dtype=np.float64).astype(np.float32).astype(np.float64)
    norm = make_normalizer(stats)
    batch = {"state": {"default": torch.as_tensor(p14).unsqueeze(0)}}
    out = norm.forward(batch)["state"]["default"][0].numpy()
    p1_linear = resid.astype(np.float64) / std32
    p1_expected = np.clip(p1_linear, -5.0, 5.0).astype(np.float32)
    p1_diff = float(np.max(np.abs(out[8:] - p1_expected)))
    assert p1_diff <= 1e-5, f"P1 normalized force != clip(residual/std32): {p1_diff}"
    mean = np.asarray(calib["mean"], dtype=np.float64)
    std = np.asarray(calib["std"], dtype=np.float64)
    p2_linear = (raw6.astype(np.float64) - mean) / std
    separation = float(np.max(np.abs(p1_linear - p2_linear)))
    assert separation > 1e-3, "P1 and P2 normalizations coincide; fixture cannot separate them"
    detail = {"fixture": "CPU fixture (arange(18) force, stubbed base8) -- NOT a GPU result",
              "p2_select_channels_raw": True, "p2_force_stream_raw": True, "scale": 1.0,
              "p1_tare_bit_exact": True,
              "p1_normalized_equals_clip_residual_over_std32": True,
              "p1_vs_p2_max_abs_separation": separation,
              "p1_expected": p1_expected.tolist(), "p2_linear": p2_linear.tolist()}
    return detail, p1_diff


def check_roundtrip(stats):
    entry = stats["state"]["default"]
    gmin = entry["global_min"].double().numpy()
    gmax = entry["global_max"].double().numpy()
    amin = stats["action"]["default"]["global_min"].double().numpy()
    amax = stats["action"]["default"]["global_max"].double().numpy()
    std32 = gmax[8:]
    norm = make_normalizer(stats)

    x_state = ((gmin + gmax) / 2.0).astype(np.float32)
    x_state[8:] = (np.array([0.0, 0.5, -0.5, 0.25, -0.25, 0.9]) * std32).astype(np.float32)
    x_action = ((amin + amax) / 2.0).astype(np.float32)
    x_action[6] = 0.5
    batch = {"state": {"default": torch.as_tensor(x_state).unsqueeze(0)},
             "action": {"default": torch.as_tensor(x_action).unsqueeze(0)}}
    forward = norm.forward({"state": {"default": batch["state"]["default"].clone()},
                            "action": {"default": batch["action"]["default"].clone()}})
    assert tuple(forward["state"]["default"].shape) == (1, 14)
    assert tuple(forward["action"]["default"].shape) == (1, 7)
    back = norm.backward(forward)
    state_diff = float((back["state"]["default"][0] - batch["state"]["default"][0]).abs().max())
    action_diff = float((back["action"]["default"][0] - batch["action"]["default"][0]).abs().max())
    assert state_diff <= 1e-5 and action_diff <= 1e-5, (state_diff, action_diff)
    assert torch.isfinite(forward["state"]["default"]).all()
    assert torch.isfinite(forward["action"]["default"]).all()

    edge = torch.zeros((1, 14), dtype=torch.float32)
    edge[0, 8:] = torch.as_tensor(std32, dtype=torch.float32)
    edge_out = norm.forward({"state": {"default": edge}})["state"]["default"][0, 8:]
    assert torch.allclose(edge_out, torch.ones(6), atol=1e-6), edge_out
    edge[0, 8:] = -torch.as_tensor(std32, dtype=torch.float32)
    edge_out = norm.forward({"state": {"default": edge}})["state"]["default"][0, 8:]
    assert torch.allclose(edge_out, -torch.ones(6), atol=1e-6), edge_out
    edge[0, 8:] = 0.0
    edge_out = norm.forward({"state": {"default": edge}})["state"]["default"][0, 8:]
    assert torch.allclose(edge_out, torch.zeros(6), atol=1e-6), edge_out

    over = torch.zeros((1, 14), dtype=torch.float32)
    over[0, 8:] = 6.0 * torch.as_tensor(std32, dtype=torch.float32)
    over[0, 0] = float(gmax[0] + 10.0 * (gmax[0] - gmin[0]))
    over_out = norm.forward({"state": {"default": over}})["state"]["default"][0]
    assert torch.equal(over_out[8:], torch.full((6,), 5.0)), over_out
    assert float(over_out[0]) == 5.0, over_out
    under = torch.zeros((1, 14), dtype=torch.float32)
    under[0, 8:] = -6.0 * torch.as_tensor(std32, dtype=torch.float32)
    under_out = norm.forward({"state": {"default": under}})["state"]["default"][0]
    assert torch.equal(under_out[8:], torch.full((6,), -5.0)), under_out

    detail = {"roundtrip_state_max_abs_diff": state_diff,
              "roundtrip_action_max_abs_diff": action_diff,
              "force_band_maps_to_pm1": True, "zero_maps_to_zero": True,
              "clip_at_plus5_exact": True, "clip_at_minus5_exact": True,
              "no_nan_inf": True, "shapes": {"state": [1, 14], "action": [1, 7]}}
    return detail, max(state_diff, action_diff)


def check_processor(stats, calib):
    from hydra.utils import instantiate

    cfg, mappings = build_cfg()
    assert int(cfg.data.train.processor.proprio_output_dim) == 14
    assert int(cfg.data.train.processor.action_output_dim) == 7
    assert str(cfg.data.train.pretrained_norm_stats) == str(RECONSTRUCTED)
    processor = instantiate(cfg.data.train.processor).eval()
    assert isinstance(processor, FastWAMProcessor)
    processor.set_normalizer_from_stats(stats)
    assert processor.proprio_output_dim == 14
    assert processor.action_output_dim == 7
    assert processor.norm_default_mode == "min/max"
    state_norm = processor.normalizer.normalizers["state"]["default"]
    action_norm = processor.normalizer.normalizers["action"]["default"]
    assert state_norm.mode == "min/max" and action_norm.mode == "min/max"
    assert int(processor.shape_meta["state"][0]["raw_shape"]) == 14
    assert int(processor.shape_meta["action"][0]["raw_shape"]) == 7

    _obs, _base8, p14, _raw6, resid = fixture_proprio14(calib)
    # The exact call sequence infer_a3_worker.normalize_proprio uses, through
    # the processor's own methods (no bridge import needed on CPU).
    batch = {"state": {"default": torch.as_tensor(p14).unsqueeze(0)}}
    batch = processor.action_state_transform(batch)
    batch = processor.normalizer.forward(batch)
    out = batch["state"]["default"]
    assert tuple(out.shape) == (1, 14)
    std32 = np.asarray(calib["std"], dtype=np.float64).astype(np.float32).astype(np.float64)
    expected = np.clip(resid.astype(np.float64) / std32, -5.0, 5.0).astype(np.float32)
    diff = float(np.max(np.abs(out[0, 8:].numpy() - expected)))
    assert diff <= 1e-5, diff
    detail = {"processor_target": cfg.data.train.processor._target_,
              "norm_default_mode": processor.norm_default_mode,
              "state_normalizer_mode": state_norm.mode,
              "action_normalizer_mode": action_norm.mode,
              "proprio_output_dim": 14, "action_output_dim": 7,
              "pretrained_norm_stats": str(RECONSTRUCTED),
              "config_path_mappings": mappings,
              "transform_plus_forward_force_diff": diff}
    return detail, diff


def run_check(name, fn):
    try:
        detail, max_abs_diff = fn()
        return {"name": name, "passed": True, "detail": detail,
                "max_abs_diff": float(max_abs_diff)}
    except (Exception, SystemExit) as exc:
        return {"name": name, "passed": False,
                "detail": f"{type(exc).__name__}: {exc}",
                "max_abs_diff": None}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    args = ap.parse_args(argv)

    print(f"NORMALIZER_IDENTITY={NORMALIZER_IDENTITY}", flush=True)
    print(f"original_stats_sha256={ORIGINAL_STATS_SHA256}", flush=True)

    for path in (RECONSTRUCTED, ORIGINAL_8D, CALIB_PATH):
        if not Path(path).is_file():
            raise SystemExit(f"required asset missing: {path}")
    recon_sha = sha256_file(RECONSTRUCTED)
    orig_sha = sha256_file(ORIGINAL_8D)
    calib_sha = sha256_file(CALIB_PATH)
    if recon_sha != RECONSTRUCTED_SHA256:
        raise SystemExit(f"reconstructed stats sha256 {recon_sha} != approved {RECONSTRUCTED_SHA256}")
    if orig_sha != ORIGINAL_8D_SHA256:
        raise SystemExit(f"original a3_fw stats sha256 {orig_sha} != {ORIGINAL_8D_SHA256}")
    if calib_sha != FORCE_CALIB_SHA256:
        raise SystemExit(f"force calib sha256 {calib_sha} != {FORCE_CALIB_SHA256}")

    calib = json.loads(CALIB_PATH.read_text())
    force_stream.check_calib(calib, where=str(CALIB_PATH))
    recon_raw = json.loads(RECONSTRUCTED.read_text())
    orig_raw = json.loads(ORIGINAL_8D.read_text())
    stats = load_dataset_stats_from_json(str(RECONSTRUCTED))
    orig_stats = load_dataset_stats_from_json(str(ORIGINAL_8D))

    checks = [
        run_check("proprio_first8_verbatim_original",
                  lambda: check_proprio_first8(recon_raw, orig_raw, stats, orig_stats)),
        run_check("force_band_pm_std32_and_calib_gate",
                  lambda: check_force_band(stats, calib, RECONSTRUCTED)),
        run_check("action7_verbatim_original",
                  lambda: check_action(recon_raw, orig_raw, stats, orig_stats)),
        run_check("p1_tare_vs_p2_raw_not_mixed",
                  lambda: check_p1_p2(calib, stats)),
        run_check("normalizer_roundtrip_band_clip",
                  lambda: check_roundtrip(stats)),
        run_check("processor_training_normalization",
                  lambda: check_processor(stats, calib)),
    ]
    verdict = "PASS" if all(c["passed"] for c in checks) else "FAIL"
    report = {
        "NORMALIZER_IDENTITY": NORMALIZER_IDENTITY,
        "original_stats_sha256": ORIGINAL_STATS_SHA256,
        "reconstructed_stats_sha256": recon_sha,
        "original_a3_fw_stats_sha256": orig_sha,
        "force_calib_sha256": calib_sha,
        "generated_utc": time.strftime("%FT%TZ", time.gmtime()),
        "scope": ("Req3 CPU normalization contract verification; no GPU, no simulator, "
                  "no merged weights; diagnosis under traceable reconstructed statistics, "
                  "NOT proof of original normalizer byte identity"),
        "inputs": {"reconstructed": str(RECONSTRUCTED), "original_8d": str(ORIGINAL_8D),
                   "force_calib": str(CALIB_PATH),
                   "force_arrays_optional": str(FORCE_ARRAYS)},
        "fw_import": FW_IMPORT_FACTS,
        "checks": checks,
        "verdict": verdict,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str))

    for c in checks:
        print(f"[{'PASS' if c['passed'] else 'FAIL'}] {c['name']} "
              f"max_abs_diff={c['max_abs_diff']}", flush=True)
        if not c["passed"]:
            print(f"       {c['detail']}", flush=True)
    print(f"CPU_ACCEPTANCE verdict={verdict} report={out_path} "
          f"NORMALIZER_IDENTITY={NORMALIZER_IDENTITY} "
          f"original_stats_sha256={ORIGINAL_STATS_SHA256}", flush=True)
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except BaseException:
        traceback.print_exc()
        sys.exit(2)