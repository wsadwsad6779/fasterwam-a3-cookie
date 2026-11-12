#!/usr/bin/env python3
"""P0 causal ordering: in one common unit, which left the training band first -- input or output?

The batch report collapsed this into two scalars (`action_absmax`, "proprio out of band") and scalars
cannot answer the question, because the two numbers live in different units. Everything here is pushed
into NORMALISED space, the only space where the corpus and the network output are comparable:

  * the min/max band in `a3_fw_v1g/dataset_stats.json` maps the corpus onto exactly [-1, +1], and the
    training-side `forward()` clamps at +-5, so "in band" means |x| <= 1 and "even the clamp was
    exceeded" means |x| > 5;
  * `actions_7d_from_model` is already in that space;
  * `proprio14` is raw, so it is mapped with the same scale/offset the trainer uses.

Per replan chunk, in time order, it prints the largest normalised action the network emitted next to
how far outside [-1, +1] the proprio conditioning of the same chunk already was.

Reads only the files the run shipped. No torch, no model, no MuJoCo.
"""
import json
import sys
from pathlib import Path

import numpy as np

STATS = Path("/workspace/a3_data/a3_fw_v1g/dataset_stats.json")
RECORD = Path("/workspace/_v1_stage/dev/out40k/record_diag/_record")
SDIM = ["d_x", "d_y", "d_z", "d_rx", "d_ry", "d_rz", "grip"]
PDIM = ["eef_x", "eef_y", "eef_z", "aa_x", "aa_y", "aa_z", "fin_l", "fin_r",
        "f_fx", "f_fy", "f_fz", "f_padL", "f_padR", "f_grip"]
EDGE = 1.0
CLAMP = 5.0
STEPS_PER_CHUNK = 10
PRED = 32


def band(st):
    lo = np.asarray(st["global_min"], dtype=np.float64)
    hi = np.asarray(st["global_max"], dtype=np.float64)
    scale = 2.0 / (hi - lo)
    offset = -1.0 - scale * lo
    return lo, hi, scale, offset


def raw_at(norm_value, scale, offset):
    return (norm_value - offset) / scale


def main():
    st = json.loads(STATS.read_text())
    alo, ahi, asu, aso = band(st["action"]["default"])
    slo, shi, psu, pso = band(st["state"]["default"])
    print("== 0. the two bands, as the trainer sees them ==")
    print("  action: corpus maps onto [%.6f, %.6f] in normalised space" %
          (float((alo * asu + aso).min()), float((ahi * asu + aso).max())))
    print("  state : corpus maps onto [%.6f, %.6f] in normalised space" %
          (float((slo * psu + pso).min()), float((shi * psu + pso).max())))
    print("  so a normalised action of 1.0 is the corpus edge, and the +-5 clamp is %sx the corpus edge"
          % round(CLAMP / EDGE, 1))
    print("  raw value at the normalised corpus edge and at the clamp, per action dim:")
    for j, nm in enumerate(SDIM):
        print("    %-6s edge=[%-9s, %-9s]  |x|=5 -> raw=%s" %
              (nm, round(float(alo[j]), 5), round(float(ahi[j]), 5),
               round(float(raw_at(CLAMP, asu, aso)[j]), 5)))

    for d in sorted(RECORD.glob("seed*")):
        af = d / ("%s_step40000_actions.npz" % d.name)
        sf = d / ("%s_step40000_steps.npz" % d.name)
        if not (af.is_file() and sf.is_file()):
            print("  %s: missing actions or steps npz" % d.name)
            continue
        za = np.load(af, allow_pickle=False)
        zs = np.load(sf, allow_pickle=False)
        if "actions_7d_from_model" not in za.files or "proprio14" not in zs.files:
            print("  %s: keys %s / %s" % (d.name, sorted(za.files), sorted(zs.files)))
            continue
        act = np.asarray(za["actions_7d_from_model"], dtype=np.float64)
        pro = np.asarray(zs["proprio14"], dtype=np.float64)
        pn = pro * psu + pso
        print("== %s: %d action rows (%d chunks x %d), %d proprio frames ==" %
              (d.name, act.shape[0], act.shape[0] // PRED, PRED, pro.shape[0]))
        print("   whole-episode, normalised space:")
        print("    action absmax   =%s" % [round(float(x), 2) for x in np.abs(act).max(0)])
        print("    action frac>1   =%s" % [round(float(x), 3) for x in (np.abs(act) > EDGE).mean(0)])
        print("    action frac>5   =%s" % [round(float(x), 3) for x in (np.abs(act) > CLAMP).mean(0)])
        print("    proprio absmax  =%s" % [round(float(x), 1) for x in np.abs(pn).max(0)])
        print("    proprio frac>1  =%s" % [round(float(x), 3) for x in (np.abs(pn) > EDGE).mean(0)])
        print("    proprio frac>5  =%s" % [round(float(x), 3) for x in (np.abs(pn) > CLAMP).mean(0)])
        for tag, thr in (("first frame over 1", EDGE), ("first frame over 5", CLAMP)):
            hit = {}
            for i in range(pn.shape[1]):
                idx = np.flatnonzero(np.abs(pn[:, i]) > thr)
                if idx.size:
                    hit[PDIM[i]] = int(idx[0])
            print("    proprio %s: %s" % (tag, hit if hit else "none"))
        ai1 = np.flatnonzero((np.abs(act) > EDGE).any(axis=1))
        ai5 = np.flatnonzero((np.abs(act) > CLAMP).any(axis=1))
        print("    first action ROW over 1: %s (chunk %s)" %
              (int(ai1[0]) if ai1.size else -1, int(ai1[0] // PRED) if ai1.size else -1))
        print("    first action ROW over 5: %s (chunk %s)" %
              (int(ai5[0]) if ai5.size else -1, int(ai5[0] // PRED) if ai5.size else -1))

        print("   per chunk: did the network's own output break the corpus edge before the input did?")
        print("    %-6s %-7s %-10s %-8s %-10s %-9s %-8s %s" %
              ("chunk", "step", "act_absmax", "act>1", "pro_absmax", "pro>1", "pro>5",
               "proprio cols already over 1"))
        nch = act.shape[0] // PRED
        for c in range(nch):
            rows = act[c * PRED:(c + 1) * PRED]
            lo_i, hi_i = c * STEPS_PER_CHUNK, min((c + 1) * STEPS_PER_CHUNK, pn.shape[0])
            pw = pn[lo_i:hi_i]
            cols = [PDIM[i] for i in range(pn.shape[1])
                    if pw.size and float(np.abs(pw[:, i]).max()) > EDGE]
            print("    %-6d %-7d %-10.3f %-8.3f %-10.2f %-9.3f %-8.3f %s" % (
                c, lo_i, float(np.abs(rows).max()), float((np.abs(rows) > EDGE).mean()),
                float(np.abs(pw).max()) if pw.size else float("nan"),
                float((np.abs(pw) > EDGE).mean()) if pw.size else float("nan"),
                float((np.abs(pw) > CLAMP).mean()) if pw.size else float("nan"),
                ",".join(cols) if cols else "-"))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
