#!/usr/bin/env python3
"""P0 cross-check, CPU only: what distribution did the force-conditioned model actually see?

The batch reported an action excursion (44.37) and a proprio excursion, and both were read out of
arrays the harness itself computed. This script refuses to use any of my own diagnostics as the
witness: every quantity is recomputed from files that the *run* shipped -- the recorded per-step
arrays, `a3_fw_v1g/dataset_stats.json` (which is also `pretrained_norm_stats` in the node's own
config.yaml), and `a3_fw_force/arrays.npz` (the corpus the six force channels were built from).

No torch, no model load, no MuJoCo.
"""
import json
import sys
from pathlib import Path

import numpy as np

RECORD = Path("/workspace/_v1_stage/dev/out40k/record_diag/_record")
STATS = Path("/workspace/a3_data/a3_fw_v1g/dataset_stats.json")
FORCE = Path("/workspace/a3_data/a3_fw_force/arrays.npz")
CALIB = Path("/workspace/a3_data/a3_fw_force/force_calib_v1.json")
COLS = ["f_fx", "f_fy", "f_fz", "f_padL", "f_padR", "f_grip"]


def q(name, arr):
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim == 1:
        a = a[:, None]
    out = []
    for i in range(a.shape[1]):
        c = a[:, i]
        c = c[np.isfinite(c)]
        if c.size == 0:
            out.append([i, None, None, None, None])
            continue
        out.append([i, round(float(np.abs(c).max()), 4), round(float(np.median(c)), 4),
                    round(float(np.percentile(np.abs(c), 99)), 4),
                    round(float(np.percentile(np.abs(c), 90)), 4)])
    return out


def main():
    print("== 0. files this reads, and their hashes ==")
    for p in (STATS, FORCE, CALIB):
        print("  %s exists=%s size=%s" % (p, p.is_file(), p.stat().st_size if p.is_file() else None))

    st = json.loads(STATS.read_text())["state"]["default"]
    lo = np.asarray(st["global_min"], dtype=np.float64)
    hi = np.asarray(st["global_max"], dtype=np.float64)
    sd = np.asarray(st["global_std"], dtype=np.float64)
    scale = 2.0 / (hi - lo)
    offset = -1.0 - scale * lo
    print("== 1. the min/max normalizer the checkpoint was trained with (state, 14 wide) ==")
    print("  scale=%s" % [round(float(x), 5) for x in scale])
    print("  offset=%s" % [round(float(x), 5) for x in offset])
    print("  force band (+/- std by construction)=%s" % [round(float(x), 5) for x in hi[8:]])
    print("  clamp at |y|=5 -> raw residual=%s" % [round(float(5.0 / scale[i]), 4) for i in range(8, 14)])

    print("== 2. the corpus the six force channels came from ==")
    f = np.load(FORCE, allow_pickle=False)
    print("  arrays.npz keys=%s" % sorted(f.files))
    for key in ("force6", "force6_zeroed"):
        if key not in f.files:
            print("  %s MISSING" % key)
            continue
        a = f[key]
        print("  %s shape=%s" % (key, a.shape))
        for row, cname in zip(q("corpus", a), COLS):
            i = row[0]
            resid_std = float(sd[8 + i])
            print("    %-7s absmax=%-10s median=%-10s p99=%-10s p90=%-10s /std=%s"
                  % (cname, row[1], row[2], row[3], row[4], round(row[1] / resid_std, 2)))
        print("    fraction |x| > 1 sigma = %.4f ; > 5 sigma = %.6f"
              % (float((np.abs(a) > sd[8:14]).mean()), float((np.abs(a) > 5 * sd[8:14]).mean())))

    print("== 3. what the evaluation fed the model, per recorded seed ==")
    for seed_dir in sorted(RECORD.glob("seed*")):
        steps = seed_dir / ("%s_step40000_steps.npz" % seed_dir.name)
        if not steps.is_file():
            print("  %s: no steps npz" % seed_dir)
            continue
        z = np.load(steps, allow_pickle=False)
        if "proprio14" not in z.files:
            print("  %s keys=%s (no proprio14)" % (seed_dir.name, sorted(z.files)))
            continue
        p = np.asarray(z["proprio14"], dtype=np.float64)
        print("  %s proprio14 shape=%s keys=%s" % (seed_dir.name, p.shape, sorted(z.files)))
        for i, name in enumerate(COLS):
            c = p[:, 8 + i]
            c = c[np.isfinite(c)]
            sig = float(sd[8 + i])
            y = c * scale[8 + i] + offset[8 + i]
            print("    %-7s absmax=%-9.4f median=%-9.4f p99=%-9.4f sigma=%-8.4f "
                  "|x|/sigma_max=%-8.2f frac>1sig=%-6.3f frac_clamped=%-6.4f norm_max=%-8.2f"
                  % (name, float(np.abs(c).max()), float(np.median(c)),
                     float(np.percentile(np.abs(c), 99)), sig, float(np.abs(c).max()) / sig,
                     float((np.abs(c) > sig).mean()), float((np.abs(y) > 5.0).mean()),
                     float(np.clip(np.abs(y), 0, 999).max())))
        for key in ("handed_14d", "applied_joint_16d", "ik_joint_target_16d"):
            if key in z.files:
                a = np.asarray(z[key], dtype=np.float64)
                a = a.reshape(a.shape[0], -1) if a.ndim > 1 else a[:, None]
                print("    %-20s shape=%s absmax per col=%s"
                      % (key, a.shape, [round(float(x), 4) for x in np.abs(a).max(axis=0)]))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(json.dumps({"fatal": str(exc)[:300]}))
        sys.exit(1)
