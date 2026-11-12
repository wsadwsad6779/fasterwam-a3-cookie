#!/usr/bin/env python3
"""P0 close-out: which layer and which dimension does the 44.37 action excursion live in?

Every stage of the action chain was recorded by `record40k.py` into
`<seed>_step40000_actions.npz` -- `actions_7d_from_model` (the raw network output, still in
normalised space), `actions_7d_denormalised` (after `denormalize_action`),
`actions_7d_after_gripper_rewrite` (after the pm1 rewrite), `actions_14d_from_adapter` (what the env
receives before its own clip) -- and the numbers this script prints come from those arrays and from
`dataset_stats.json` / the adapter source, not from any diagnostic of mine.

It answers three separate questions that the batch report collapsed into one scalar:
  * the magnitude at each stage, per dimension, so the stage that creates the excursion is named;
  * the normalised value the network would have had to emit to produce it, through the V1G min/max
    band -- against the +-5 the training-side forward() clamps to;
  * whether the env-side clip at +-1 was already saturating, i.e. whether the excursion ever reached
    the physics at all.
"""
import json
import sys
from pathlib import Path

import numpy as np

STATS = Path("/workspace/a3_data/a3_fw_v1g/dataset_stats.json")
RECORD = Path("/workspace/_v1_stage/dev/out40k/record_diag/_record")
DIM = ["d_x", "d_y", "d_z", "d_rx", "d_ry", "d_rz", "grip"]
CLAMP = 5.0


def main():
    st = json.loads(STATS.read_text())["action"]["default"]
    lo = np.asarray(st["global_min"], dtype=np.float64)
    hi = np.asarray(st["global_max"], dtype=np.float64)
    scale = 2.0 / (hi - lo)
    offset = -1.0 - scale * lo
    print("== 0. the action band the checkpoint was trained with ==")
    print("  keys in stats[action][default]: %s" % sorted(st.keys()))
    for j, name in enumerate(DIM):
        print("    %-6s min=%-10s max=%-10s scale=%-9s offset=%-9s |x|=%.1f -> raw=%s"
              % (name, round(float(lo[j]), 5), round(float(hi[j]), 5), round(float(scale[j]), 3),
                 round(float(offset[j]), 3), CLAMP,
                 round(float((CLAMP - offset[j]) / scale[j]), 5)))

    for d in sorted(RECORD.glob("seed*")):
        f = d / ("%s_step40000_actions.npz" % d.name)
        if not f.is_file():
            print("  %s: no actions npz" % d.name)
            continue
        z = np.load(f, allow_pickle=False)
        print("== %s: keys=%s ==" % (d.name, sorted(z.files)))
        stages = [("model_out", "actions_7d_from_model"),
                  ("denorm", "actions_7d_denormalised"),
                  ("pm1", "actions_7d_after_gripper_rewrite"),
                  ("adapter", "actions_14d_from_adapter")]
        for label, key in stages:
            if key not in z.files:
                print("  %-8s MISSING (%s)" % (label, key))
                continue
            a = np.asarray(z[key], dtype=np.float64)
            print("  %-8s shape=%s" % (label, a.shape))
            if a.ndim == 3:
                a = a.reshape(-1, a.shape[-1])
            ncols = a.shape[1]
            names = DIM if ncols == 7 else (
                ["x", "y", "z", "rx", "ry", "rz", "grip",
                 "X", "Y", "Z", "RX", "RY", "RZ", "GRIP"][:ncols])
            amax = np.abs(a).max(axis=0)
            for i in range(ncols):
                nm = names[i] if i < len(names) else "col%d" % i
                line = "    [%2d] %-6s absmax=%-10.4f p99=%-10.4f med=%-9.4f" % (
                    i, nm, amax[i], np.percentile(np.abs(a[:, i]), 99), np.median(np.abs(a[:, i])))
                if ncols == 7 and i < 7 and label in ("model_out", "denorm"):
                    if label == "model_out":
                        raw = (a[:, i] - offset[i]) / scale[i]
                        line += "  | denorm-of-max=%-11.4f frac|x|>5=%-6.3f" % (
                            float(np.abs(raw).max()), float((np.abs(a[:, i]) > CLAMP).mean()))
                    else:
                        nrm = a[:, i] * scale[i] + offset[i]
                        line += "  | norm-of-max=%-11.3f frac|norm|>5=%-6.3f" % (
                            float(np.abs(nrm).max()), float((np.abs(nrm) > CLAMP).mean()))
                print(line)
        m = z["measured"] if "measured" in z.files else None
        if m is not None:
            try:
                ma = np.asarray(m, dtype=np.float64)
            except (TypeError, ValueError):
                print("  measured is not numeric: shape=%s dtype=%s" % (np.asarray(m).shape,
                                                                        np.asarray(m).dtype))
            else:
                flat = ma.reshape(-1, ma.shape[-1]) if ma.ndim >= 2 else ma[:, None]
                print("  measured shape=%s absmax per col=%s" %
                      (ma.shape, [round(float(x), 4) for x in np.abs(flat).max(0)]))
        rp = z["reset_proprio14"] if "reset_proprio14" in z.files else None
        if rp is not None:
            print("  reset_proprio14=%s" % [round(float(x), 5) for x in np.asarray(rp, np.float64)])


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
