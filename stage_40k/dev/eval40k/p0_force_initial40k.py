#!/usr/bin/env python3
"""P0 boundary check: what does the force conditioning look like at the START of an episode?

The two failure stories below this line are very different, and which one is true decides whether the
evaluation interface or the trained weights own the problem:

  * if the force columns only leave the training band after the arm has moved, the model went there
    first and the sensor input followed;
  * if they leave the band in the first control steps -- while the gripper is 0.9 m above the table
    and physically cannot be touching anything -- then the evaluation is handing the model an input
    that the training distribution never contained, at the moment the policy is most sensitive to it.

The corpus cannot show the second case by construction: `build_force_arrays.py` tares each episode on
its own first 20 frames, so inside that window the residual is the deviation from the episode's own
median. The evaluation has no such window: `make_force_calib.py` deliberately replaced it with one
global constant, the median of the per-episode medians. So the two things to measure are how much the
per-episode tares scatter around that constant, and what the residual actually is in the first frames
of the roll-outs.

Everything is expressed in the unit the model sees: the min/max normaliser of
`a3_fw_v1g/dataset_stats.json`, where the corpus spans [-1, +1] and `forward()` clamps at +-5.
Reads only; no torch, no model, no MuJoCo.
"""
import json
import sys
from pathlib import Path

import numpy as np

V1G = Path("/workspace/a3_data/a3_fw_v1g/arrays.npz")
FORCE = Path("/workspace/a3_data/a3_fw_force/arrays.npz")
CALIB = Path("/workspace/a3_data/a3_fw_force/force_calib_v1.json")
STATS = Path("/workspace/a3_data/a3_fw_v1g/dataset_stats.json")
RECORD = Path("/workspace/_v1_stage/dev/out40k/record_diag/_record")
WIN = 20
FC = ["f_fx", "f_fy", "f_fz", "f_padL", "f_padR", "f_grip"]


def main():
    cal = json.loads(CALIB.read_text())
    tare = np.asarray(cal["zero_load_ref_median"], dtype=np.float64)
    st = json.loads(STATS.read_text())["state"]["default"]
    lo = np.asarray(st["global_min"], float)
    hi = np.asarray(st["global_max"], float)
    psu = 2.0 / (hi - lo)
    pso = -1.0 - psu * lo

    def norm6(res6):
        r = np.asarray(res6, dtype=np.float64)
        if r.ndim == 1:
            r = r[None, :]
        return r * psu[8:14] + pso[8:14]

    print("== 0. the band in the unit the model reads ==")
    print("  corpus normalised range (by construction) = [-1, 1]; forward() clamps at +-5")
    print("  raw residual that reaches the clamp, per channel: y=+5 -> %s N, y=-5 -> %s N" %
          ([round(float((5.0 - pso[8 + i]) / psu[8 + i]), 4) for i in range(6)],
           [round(float((-5.0 - pso[8 + i]) / psu[8 + i]), 4) for i in range(6)]))
    print("  ...and the raw residual at the corpus edge (+-1), per channel = %s N" %
          [round(float((1.0 - pso[8 + i]) / psu[8 + i]), 4) for i in range(6)])
    print("  1 sigma (calibration) = %s N" % [round(float(x), 5) for x in cal["std"]])

    f = np.load(FORCE, allow_pickle=False)
    v = np.load(V1G, allow_pickle=False)
    raw6 = np.asarray(f["force6"], dtype=np.float64)
    ep = np.asarray(f["episode_index"], dtype=np.int64)
    ep_start = np.asarray(f["ep_start"], dtype=np.int64)
    ep_len = np.asarray(f["ep_len"], dtype=np.int64)
    p14 = np.asarray(v["proprio"], dtype=np.float64)
    n = p14.shape[0]
    starts = np.zeros(n, dtype=np.int64)
    for k, s in enumerate(ep_start):
        starts[s:s + int(ep_len[k])] = np.arange(int(ep_len[k]))

    print("== 1. how much do the per-episode tares scatter around the one constant the eval uses? ==")
    if "zero_load_ref_per_episode" not in cal:
        print("  cal keys=%s -- no per-episode tare recorded, cannot measure the scatter" %
              sorted(cal))
        per = None
    else:
        per = np.asarray(cal["zero_load_ref_per_episode"], dtype=np.float64)
    if per is None:
        pass
    elif per.ndim != 2 or per.shape[1] != 6:
        print("  zero_load_ref_per_episode shape=%s -- cannot use" % (per.shape,))
    else:
        print("  episodes=%d" % per.shape[0])
        for i, nm in enumerate(FC):
            c = per[:, i]
            d = c - tare[i]
            print("    %-7s global-const=%-10s per-episode min=%-10s max=%-10s std=%-9s "
                  "|scatter vs the constant| med=%-8s p95=%-8s max=%-8s N" %
                  (nm, round(float(tare[i]), 4), round(float(c.min()), 4), round(float(c.max()), 4),
                   round(float(c.std()), 4), round(float(np.abs(d).mean()), 4),
                   round(float(np.percentile(np.abs(d), 95)), 4), round(float(np.abs(d).max()), 4)))
        dn = np.abs(norm6(per - tare))
        print("    the same scatter in model units: |normalised| med=%s p95=%s max=%s  "
              "(1.0 = the corpus edge, 5.0 = the clamp)" %
              ([round(float(np.median(dn[:, i])), 3) for i in range(6)],
               [round(float(np.percentile(dn[:, i], 95)), 3) for i in range(6)],
               [round(float(dn[:, i].max()), 3) for i in range(6)]))
        print("    episodes whose tare alone would already exceed the clamp on some channel: %d / %d"
              % (int((dn > 5.0).any(axis=1).sum()), dn.shape[0]))

    print("== 2. inside the tare window the corpus residual is near zero by construction ==")
    win = raw6[starts < WIN] - tare
    wn = norm6(win)
    print("  rows=%d   |normalised residual| med=%s p95=%s max=%s" %
          (win.shape[0], [round(float(np.median(np.abs(wn[:, i]))), 3) for i in range(6)],
           [round(float(np.percentile(np.abs(wn[:, i]), 95)), 3) for i in range(6)],
           [round(float(np.abs(wn[:, i]).max()), 3) for i in range(6)]))
    out = p14[:, 8:14]
    on = out * psu[8:14] + pso[8:14]
    print("  the stored corpus columns (what training read), absmax normalised=%s  frac over 5=%s" %
          ([round(float(np.abs(on[:, i]).max()), 2) for i in range(6)],
           [round(float((np.abs(on[:, i]) > 5).mean()), 4) for i in range(6)]))
    print("  corpus: fraction of ALL frames with a clamped force channel=%s" %
          [round(float((np.abs(on[:, i]) > 5).mean()), 4) for i in range(6)])

    print("== 3. the roll-outs: the same channels in the first control steps ==")
    for d in sorted(RECORD.glob("seed*")):
        if not d.is_dir():
            print("  %s: not a directory" % d.name)
            continue
        cands = sorted(d.glob("*_steps.npz")) or sorted(d.glob("*.npz"))
        if not cands:
            print("  %s: no npz" % d.name)
            continue
        sf = cands[0]
        print("  file=%s" % sf.name)
        z = np.load(sf, allow_pickle=False)
        if "proprio14" not in z.files:
            print("  %s: no proprio14 (keys=%s)" % (d.name, sorted(z.files)))
            continue
        e = np.asarray(z["proprio14"], dtype=np.float64)
        en = e * psu + pso
        fn = en[:, 8:14]
        print("  %s frames=%d" % (d.name, e.shape[0]))
        print("    reset column (from the npz, raw)=%s" %
              ([round(float(x), 5) for x in e[0, 8:14]] if "reset_proprio14" not in z.files else
               [round(float(x), 5) for x in np.asarray(z["reset_proprio14"], float)[8:14]]))
        print("    first frames, raw residual per channel (should be ~0: nothing is touching yet):")
        for k in range(min(6, fn.shape[0])):
            print("      frame %-2d raw=%s  normalised=%s  clamped=%s" %
                  (k, [round(float(x), 3) for x in e[k, 8:14]],
                   [round(float(x), 2) for x in fn[k]],
                   [FC[i] for i in range(6) if abs(fn[k, i]) > 5.0]))
        over1 = np.flatnonzero((np.abs(fn) > 1.0).any(axis=1))
        over5 = np.flatnonzero((np.abs(fn) > 5.0).any(axis=1))
        print("    first frame any force channel over +-1: %s ; over the +-5 clamp: %s" %
              (int(over1[0]) if over1.size else -1, int(over5[0]) if over5.size else -1))
        print("    frames 0-19: frac clamped=%s ; frames 20-59: frac clamped=%s ; all: frac clamped=%s"
              % ([round(float((np.abs(fn[:WIN, i]) > 5).mean()), 3) for i in range(6)],
                 [round(float((np.abs(fn[WIN:60, i]) > 5).mean()), 3) for i in range(6)],
                 [round(float((np.abs(fn[:, i]) > 5).mean()), 3) for i in range(6)]))
        print("    eef_z over the first 20 frames: min=%s max=%s (corpus eef_z in [0.7638, 0.9200])"
              % (round(float(e[:WIN, 2].min()), 4), round(float(e[:WIN, 2].max()), 4)))
        print("    finger joints over the first 20 frames: fin_l=%s..%s fin_r=%s..%s" %
              (round(float(e[:WIN, 6].min()), 4), round(float(e[:WIN, 6].max()), 4),
               round(float(e[:WIN, 7].min()), 4), round(float(e[:WIN, 7].max()), 4)))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
