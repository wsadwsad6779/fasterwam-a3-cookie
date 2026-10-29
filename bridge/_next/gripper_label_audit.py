"""Audit the gripper channel of the A3 training arrays. Numpy only.

`prepare_a3_arrays.py` builds the 7th action from the *measured* opening:

    g[t] = +1 if opening[t+1] < opening[t] - 1e-4     (close)
    g[t] = -1 if opening[t+1] > opening[t] + 1e-4     (open)
    g[t] = g[t-1] otherwise                            (hold the last label)

When the pads contact the cookie the measured opening rebounds, so the "hold"
rule latches the wrong sign; while the gripper sits still, the label is decided
by the last transient instead of by the command. The stored `joint_cmd`
(`action[:, 7]` in the parquet) is the actual gripper command, so it can judge
the label.

    python3 gripper_label_audit.py ../../../WAM/a3_handoff_202610271/data/arrays.npz
"""
import json
import sys
from pathlib import Path

import numpy as np

MAX_GRIPPER_STEP = 0.08


def audit(arrays_path):
    d = np.load(arrays_path)
    act, jc = d["action"], d["joint_cmd"]
    starts, lens = d["ep_start"], d["ep_len"]

    labels, deltas, cmds = [], [], []
    for s, L in zip(starts, lens):
        labels.append(act[s:s + L - 1, 6])
        deltas.append(np.diff(jc[s:s + L]))
        cmds.append(jc[s:s + L - 1])
    labels = np.concatenate(labels)
    deltas = np.concatenate(deltas)
    cmds = np.concatenate(cmds)

    moving = np.abs(deltas) > 1e-4
    static = ~moving
    holding = static & (cmds < 0.50)
    fully_open = static & (cmds > 0.90)

    # LIBERO sign: +1 close, -1 open. The command is a width, so a decreasing
    # command means closing; agreement therefore needs the opposite signs.
    agrees = np.sign(labels) == -np.sign(deltas)
    wrong = (moving & ~agrees) | (holding & (labels < 0)) | (fully_open & (labels > 0))

    cont = np.clip(-deltas / MAX_GRIPPER_STEP, -1.0, 1.0)  # LIBERO convention

    return {
        "frames": int(labels.size),
        "moving_pct": round(100.0 * moving.mean(), 2),
        "moving_label_matches_command_pct": round(100.0 * agrees[moving].mean(), 2),
        "static_holding_frames": int(holding.sum()),
        "static_holding_labeled_open_pct": round(100.0 * (labels[holding] < 0).mean(), 2),
        "static_open_frames": int(fully_open.sum()),
        "static_open_labeled_close_pct": round(100.0 * (labels[fully_open] > 0).mean(), 2),
        "wrong_label_pct": round(100.0 * wrong.mean(), 2),
        "command_abs_delta_p50": round(float(np.percentile(np.abs(deltas), 50)), 5),
        "command_abs_delta_p95": round(float(np.percentile(np.abs(deltas), 95)), 5),
        "max_abs_delta": round(float(np.abs(deltas).max()), 5),
        "continuous_label_hold_pct": round(100.0 * float((np.abs(cont) < 0.02).mean()), 2),
        "continuous_label_saturated_pct": round(100.0 * float((np.abs(cont) > 0.99).mean()), 2),
    }


def main():
    path = Path(sys.argv[1] if len(sys.argv) > 1 else
                "../../../WAM/a3_handoff_202610271/data/arrays.npz")
    res = audit(path)
    for k, v in res.items():
        print(f"{k:40s} {v}")
    out = path.with_name("gripper_label_audit.json")
    out.write_text(json.dumps(res, indent=2))
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
