"""Read the action normalizer that is actually installed, and what it does to a big number.

Point 1(c) of the current work order asks whether "the action normalizer in use" equals the
training statistics. `eval40k.py:734-735,775` does load `/workspace/a3_data/a3_fw_v1g/dataset_stats.json`
and call `set_normalizer_from_stats`, and `run40k/config.yaml` says `norm_default_mode: min/max`,
`use_stepwise_action_norm: false`, `norm_exception_mode: null`, `action_state_transforms: null`,
`action_output_dim: 7` -- so the map is `min + (x+1)(max-min)/2` per column with no rotation,
relative-action or stepwise transform to invert, and `ConcatLeftAlign` pads nothing because
`action_output_dim` already equals the 7-wide band.

This script proves that on the live object instead of arguing from the config text, and then prints
the table the batch cannot give from disk: for each column, the normalised value the model would
have had to output to produce that seed's `action_absmax`. No model is loaded; no GPU is touched.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import eval40k as E

BAND_FILE = Path("/workspace/a3_data/a3_fw_v1g/dataset_stats.json")
DIM_NAMES = ("d_x", "d_y", "d_z", "d_rx", "d_ry", "d_rz", "gripper")
UNIT_NOTE = ("adapter: dims 0-2 are LIBERO OSC delta units (1.0 = 0.05 m) and are doubled into A3 "
             "units (1.0 = 0.025 m); dims 3-5 are halved world rotvec then divided by 0.1 rad; "
             "dim 6 is binarised; the result is clipped elementwise to [-1, 1]")


def main() -> None:
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from fasterwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json

    doc = OmegaConf.load(str(E.DEFAULT_CFG))
    stats = load_dataset_stats_from_json(str(BAND_FILE))
    processor = instantiate(doc.data.train.processor).eval()
    processor.set_normalizer_from_stats(stats)
    nrm = processor.normalizer.normalizers["action"]["default"]

    print("== live action normalizer ==")
    print("  class            %s" % type(nrm).__name__)
    print("  mode             %r" % nrm.mode)
    print("  output_min/max   %s / %s   range_tol %s"
          % (nrm.output_min, nrm.output_max, nrm.range_tol))
    print("  stats keys       %s" % sorted(nrm.stats))
    print("  processor        action_output_dim=%s proprio_output_dim=%s delta_mask=%s "
          "transforms=%s merger=%s"
          % (processor.action_output_dim, processor.proprio_output_dim,
             None if processor.delta_action_dim_mask is None
             else processor.delta_action_dim_mask["default"].tolist(),
             processor.action_state_transforms,
             type(processor.action_state_merger).__name__))

    lo = nrm.stats["min"].double().numpy()
    hi = nrm.stats["max"].double().numpy()
    scale = nrm.scale.double().numpy()
    offset = nrm.offset.double().numpy()
    expect_scale = 2.0 / (hi - lo)
    print("\n== per-column map (min/max mode) ==")
    print("the scale the constructor stored is computed in float32, so the tolerance here is a "
          "relative one; the property that must hold exactly is backward(-1)=min, backward(+1)=max")
    print("%-8s %-10s %-10s %-14s %-14s %-14s %s"
          % ("dim", "min", "max", "scale(live)", "scale(float64)", "rel_diff", "backward(-1)/backward(+1)"))
    for j in range(7):
        y_lo = float(torch_minus(offset[j], -1.0) / scale[j])
        y_hi = float(torch_minus(offset[j], 1.0) / scale[j])
        rel = abs(scale[j] - expect_scale[j]) / max(abs(expect_scale[j]), 1e-12)
        print("%-8s %-10.6f %-10.6f %-14.6f %-14.6f %-14.2e %.6f / %.6f"
              % (DIM_NAMES[j], lo[j], hi[j], scale[j], expect_scale[j], rel, y_lo, y_hi))
    print("  backward(-1) must equal min and backward(+1) must equal max for every column")
    ok_edges = all(abs(float((1.0 - offset[j]) / scale[j]) - hi[j]) < 1e-6
                   and abs(float((-1.0 - offset[j]) / scale[j]) - lo[j]) < 1e-6 for j in range(7))
    print("  all seven columns pass: %s" % ok_edges)

    # round trip through the forward map the training side used
    grid = np.linspace(0.0, 1.0, 11, dtype=np.float64)[None, :]
    x = lo[:, None] + (hi - lo)[:, None] * grid
    scale_t = nrm.scale.double().numpy()[:, None]
    offset_t = nrm.offset.double().numpy()[:, None]
    norm = np.clip(x * scale_t + offset_t, -5.0, 5.0)
    back = (norm - offset_t) / scale_t
    print("\nround trip max abs error inside the band: %.2e" % np.abs(back - x).max())

    print("\n== what the live map does to an out-of-band normalised output ==")
    print("%-8s %s" % ("dim", " ".join("x=%-6s" % v for v in (1, 2, 5, 10, 50, 100))))
    for j in range(7):
        vals = [float((v - offset[j]) / scale[j]) for v in (1, 2, 5, 10, 50, 100)]
        print("%-8s %s" % (DIM_NAMES[j], " ".join("%-8.3f" % v for v in vals)))

    print("\n== normalised value required to yield each recorded action_absmax ==")
    rec_root = Path("/workspace/_v1_stage/dev/out40k/batch_eval_20seed1x/fixed/step_040000")
    rows = []
    for f in sorted(rec_root.glob("episode_seed*.json")):
        d = json.loads(f.read_text())
        rows.append((d.get("seed"), d.get("action_absmax_max")))
    print("%-6s %-10s %s" % ("seed", "absmax", "required normalised x per dim (0..5)"))
    for seed, absmax in rows:
        if absmax is None:
            continue
        need = [float(absmax * scale[j] + offset[j]) for j in range(6)]
        print("%-6s %-10.4f %s" % (seed, absmax, " ".join("%-9.1f" % v for v in need)))
    print("\nnote: `action_absmax` is taken after the pm1 gripper rewrite, so column 6 is pinned to "
          "+-1 and can never be the max; every absmax above 1.0 is a delta column.")
    print("note: %s" % UNIT_NOTE)


def torch_minus(offset, x):
    return x - offset


if __name__ == "__main__":
    main()
