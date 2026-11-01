"""Low-intervention C0 metrics from the eval-only cookie truth log.

Reads the executed npz written by `run_matrix.run_one` / `eval_c0_diag.py` and
reports, per policy seed, the four numbers the C0 table uses:

  * `max_cookie_lift_mm`   -- max over cookies of (z - z at reset)
  * `stable_follow_steps`  -- longest run of consecutive steps in which one
    cookie is lifted (>5 mm), both finger pads are in contact, and its
    displacement relative to the TCP stays under 2 mm per step
  * `transport_xy_mm`      -- horizontal distance a cookie covers while lifted

`cookies_in_target` and `episode_success` come from the harness summary itself;
they are not recomputed here so that there is only one scoring path.
Cookie poses are read only for scoring; nothing here feeds the policy.

    python3 analyze_lift.py --glob '/workspace/a3_eval/r3_c0_diag_600/rep0/lora_C0_*.npz'
"""
import argparse
import glob
import json

import numpy as np


def longest(mask):
    best = run = 0
    for x in mask:
        run = run + 1 if x else 0
        best = max(best, run)
    return int(best)


def score(path):
    m = np.load(path)
    ck, eef = m["cookie_xyz"], m["eef_xyz"]
    dz = ck[:, :, 2] - m["cookie_xyz0"][:, 2]
    lifted = dz > 0.005
    contact = (m["pad_force_inner"] > 2.0) & (m["pad_force_outer"] > 2.0)
    rel = ck - eef[:, None, :]
    slip = np.zeros(dz.shape, dtype=bool)
    slip[1:] = np.linalg.norm(np.diff(rel, axis=0), axis=2) < 0.002
    good = lifted & contact[:, None] & slip
    per_cookie = [longest(good[:, c]) for c in range(good.shape[1])]
    # horizontal distance a cookie travels while it is off the pad (>5 mm)
    transport = 0.0
    for c in range(dz.shape[1]):
        idx = np.nonzero(lifted[:, c])[0]
        if idx.size:
            p = ck[idx[0]:idx[-1] + 1, c, :2]
            transport = max(transport, float(np.linalg.norm(p[-1] - p[0])))
    return {
        "seed": int(str(path).rsplit("_", 1)[1][: -len(".npz")]),
        "max_cookie_lift_mm": round(float(dz.max() * 1000.0), 1),
        "stable_follow_steps": max(per_cookie),
        "per_cookie_follow": per_cookie,
        "transport_xy_mm": round(transport * 1000.0, 1),
        "grasp_frames_frac": round(float(good.any(axis=1).mean()), 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rows = [score(p) for p in sorted(glob.glob(args.glob))]
    print(json.dumps(rows, indent=1), flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
