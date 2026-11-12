"""Audit all 18 A3 force channels, then compare candidate force-input sets.

Read-only. Writes a JSON report plus a printable table.
Causal-calibration note: the load reference is the median of the first
`LOAD_FRAMES` frames of each episode, reduced to one global per-channel value,
because a fixed global constant is the only thing inference can use on frame 0.
"""
import json
import sys

import numpy as np

ARR = "/workspace/a3_data/a3_fw_force/arrays.npz"
OUT = "/workspace/_stage/chan18.json"
LOAD_FRAMES = 20
CLAMP = 5.0

NAMES = [
    "L_wrist_fx", "L_wrist_fy", "L_wrist_fz",
    "L_wrist_tx", "L_wrist_ty", "L_wrist_tz",
    "R_wrist_fx", "R_wrist_fy", "R_wrist_fz",
    "R_wrist_tx", "R_wrist_ty", "R_wrist_tz",
    "L_pad_inner", "L_pad_outer", "R_pad_inner", "R_pad_outer",
    "L_grip_act", "R_grip_act",
]
UNIT = ["N"] * 3 + ["Nm"] * 3 + ["N"] * 3 + ["Nm"] * 3 + ["N"] * 4 + ["N"] * 2

CANDIDATES = {
    "C1_v6_current": [0, 1, 2, 12, 13, 16],
    "C2_l_wrist_wrench": [0, 1, 2, 3, 4, 5],
    "C3_wrench_plus_pads": [0, 1, 2, 3, 4, 5, 12, 13, 16],
    "C4_both_wrists_pads": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
    "C5_wrench_both_pads": [0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17],
}


def pct(a, q):
    return float(np.percentile(a, q))


def main():
    arr = np.load(ARR, allow_pickle=True)
    f = np.asarray(arr["force18"], dtype=np.float64)
    n, c = f.shape
    assert c == 18, c
    ep_start = np.asarray(arr["ep_start"]).astype(np.int64)
    ep_len = np.asarray(arr["ep_len"]).astype(np.int64)
    opening = np.asarray(arr["opening"], dtype=np.float64)
    act = np.asarray(arr["action"], dtype=np.float64)

    # global load reference: median over each episode's first LOAD_FRAMES frames
    load_rows = []
    for s, L in zip(ep_start, ep_len):
        k = min(LOAD_FRAMES, int(L))
        load_rows.append(f[s:s + k])
    load_pool = np.concatenate(load_rows, axis=0)
    Z_global = np.median(load_pool, axis=0)
    per_ep = np.stack([np.median(x, axis=0) for x in load_rows])
    ep_spread = per_ep.std(axis=0)

    res = f - Z_global

    # contact phase proxy: left pads both below their 25th pct => pinch free space
    padL = np.maximum(f[:, 12], f[:, 13])
    thr = pct(padL, 25)
    contact = padL > thr

    # gripper-closing events from the binary action channel
    close = act[:, 6] > 0

    report = {"n_frames": int(n), "n_eps": int(len(ep_start)),
              "hz": 20, "load_frames": LOAD_FRAMES,
              "Z_global": Z_global.tolist(), "channels": []}

    for i in range(18):
        raw = f[:, i]
        r = res[:, i]
        sd = float(r.std())
        safe = sd if sd > 1e-9 else 1.0
        norm = np.clip(r / safe, -CLAMP, CLAMP)
        # response of this channel between free-space and contact frames
        m_free = float(np.median(r[~contact]))
        m_cont = float(np.median(r[contact]))
        report["channels"].append({
            "idx": i, "name": NAMES[i], "unit": UNIT[i],
            "raw_median": float(np.median(raw)),
            "raw_min": float(raw.min()), "raw_max": float(raw.max()),
            "raw_p01": pct(raw, 1), "raw_p99": pct(raw, 99),
            "raw_std": float(raw.std()),
            "zero_ref": float(Z_global[i]),
            "ep_ref_spread_std": float(ep_spread[i]),
            "ep_ref_spread_in_resid_std": float(ep_spread[i] / safe),
            "resid_mean": float(r.mean()),
            "resid_median": float(np.median(r)),
            "resid_std": sd,
            "resid_p01": pct(r, 1), "resid_p99": pct(r, 99),
            "resid_absmax": float(np.abs(r).max()),
            "contact_median_shift": m_cont - m_free,
            "contact_shift_in_std": (m_cont - m_free) / safe,
            "corr_with_Lpadmax": float(np.corrcoef(r, padL)[0, 1]),
            "corr_with_close_cmd": float(np.corrcoef(r.astype(np.float64), close.astype(np.float64))[0, 1]),
            "clip_frac_at_5": float((np.abs(r / safe) > CLAMP).mean()),
            "constant_offset_in_std": float(abs(r.mean()) / safe),
            "nan": int(np.isnan(raw).sum()),
        })

    ch = report["channels"]
    # correlation between the left wrist torque triple and the left force triple
    idx_pairs = [(0, 3), (1, 4), (2, 5), (2, 16), (3, 12), (4, 12), (5, 16), (0, 6), (2, 8), (2, 14)]
    report["pairwise_corr"] = [
        {"a": NAMES[i], "b": NAMES[j], "r": float(np.corrcoef(res[:, i], res[:, j])[0, 1])}
        for i, j in idx_pairs
    ]

    for key, cols in CANDIDATES.items():
        tot_clip = float(np.mean([ch[i]["clip_frac_at_5"] for i in cols]))
        worst_off = float(max(ch[i]["constant_offset_in_std"] for i in cols))
        min_shift = float(min(abs(ch[i]["contact_shift_in_std"]) for i in cols))
        med_shift = float(np.median([abs(ch[i]["contact_shift_in_std"]) for i in cols]))
        report.setdefault("candidates", {})[key] = {
            "dims": len(cols), "proprio_total": 8 + len(cols),
            "clip_frac_mean": tot_clip,
            "worst_constant_offset_std": worst_off,
            "min_abs_contact_shift_std": min_shift,
            "median_abs_contact_shift_std": med_shift,
            "cols": cols, "names": [NAMES[i] for i in cols],
        }

    with open(OUT, "w") as fh:
        json.dump(report, fh, indent=1, sort_keys=False)

    print("idx name                unit  zero_ref    resid_std  absmax   off/std  contact_shift/std  corrLpad  clip5")
    for x in ch:
        print("%3d  %-18s %-4s %10.3f %10.4f %8.3f %8.3f %12.3f %10.3f %8.5f" % (
            x["idx"], x["name"], x["unit"], x["zero_ref"], x["resid_std"],
            x["resid_absmax"], x["constant_offset_in_std"], x["contact_shift_in_std"],
            x["corr_with_Lpadmax"], x["clip_frac_at_5"]))
    print()
    for k, v in report["candidates"].items():
        print("%-22s dims=%2d proprio=%2d clip=%.5f worst_off=%.2f min_shift=%.3f med_shift=%.3f" % (
            k, v["dims"], v["proprio_total"], v["clip_frac_mean"],
            v["worst_constant_offset_std"], v["min_abs_contact_shift_std"],
            v["median_abs_contact_shift_std"]))
    print()
    for p in report["pairwise_corr"]:
        print("corr %-14s %-14s %+.3f" % (p["a"], p["b"], p["r"]))
    print("WROTE " + OUT)
    print("DONE_CHAN18")


if __name__ == "__main__":
    sys.exit(main())
