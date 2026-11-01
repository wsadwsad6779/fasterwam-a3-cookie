"""Aggregate the stage-B matrix: authority, success and oracle dependency.

Reads `<model>_<control>_<seed>.npz` written by `run_matrix.py`. Runs anywhere
(numpy only) -- it never imports mujoco or torch.

    python3 analyze_runs.py outputs/stageB

Prints, per cell, the numbers that decide the experiment; plus a self-check that
the `safety` control really is low-intervention (verbatim should be 100% unless
a safety rule fired).
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

PHASES = ("open", "preclose", "descend", "close", "hold", "release", "done", "raw")
POS_MM = 25.0
ROT_MRAD = 100.0


def per_run(path):
    d = np.load(path)
    pre, post = d["pre_14d"], d["post_14d"]
    dp = (post - pre) * POS_MM
    dr = (post - pre) * ROT_MRAD
    changed = np.any(np.abs(post - pre) > 1e-9, axis=1)
    max_out = np.maximum(d["pad_force_inner"], d["pad_force_outer"])
    out = {
        "steps": int(len(post)),
        "verbatim_pct": round(100.0 * float((~changed).mean()), 2),
        "rewrite_mm_mean": round(float(np.linalg.norm(dp[:, :3], axis=1).mean()), 3),
        "rewrite_mm_p95": round(float(np.percentile(np.linalg.norm(dp[:, :3], axis=1), 95)), 3),
        "model_motion_mm_mean": round(float(np.linalg.norm(pre[:, :3] * POS_MM, axis=1).mean()), 3),
        "rewrite_mrad_mean": round(float(np.linalg.norm(dr[:, 3:6], axis=1).mean()), 2),
        "grip_changed_pct": round(100.0 * float((np.abs(post[:, 6] - pre[:, 6]) > 1e-9).mean()), 2),
        "max_pad_force": round(float(max_out.max()), 3),
        "oracle_cookie_reads": int(d["oracle_cookie"][-1]),
        "oracle_data_reads": int(d["oracle_data"][-1]),
        "sensor_reads": int(d["oracle_sensor"][-1]),
    }
    # Per-channel authority. A staged hand-off releases one skill at a time, so
    # "verbatim 40%" is not actionable on its own: the wrapper may be rewriting
    # only the gripper, which means the model already owns the arm motion.
    delta = post - pre
    for name, sl in (("xyz", slice(0, 3)), ("rot", slice(3, 6)), ("grip", slice(6, 7))):
        touched = np.any(np.abs(delta[:, sl]) > 1e-9, axis=1)
        out[f"verbatim_{name}_pct"] = round(100.0 * float((~touched).mean()), 2)
    if "wrist_force" in d:
        out["max_wrist_force"] = round(float(np.linalg.norm(d["wrist_force"], axis=1).max()), 2)
    phase = d["phase"]
    if phase.max() >= 0:
        out["by_phase"] = {}
        for pid in sorted(set(phase.tolist())):
            if pid < 0:
                continue
            sel = phase == pid
            entry = {
                "steps": int(sel.sum()),
                "verbatim_pct": round(100.0 * float((~changed[sel]).mean()), 2),
                "rewrite_mm_mean": round(float(np.linalg.norm(dp[sel, :3], axis=1).mean()), 3),
            }
            # A capability stage (e.g. C1 = transport xyz only) can show 0%
            # whole-run xyz authority with the model already owning every
            # approach step, so authority has to be read per phase too.
            for ch, sl in (("xyz", slice(0, 3)), ("rot", slice(3, 6)),
                           ("grip", slice(6, 7))):
                touched = np.any(np.abs(delta[sel, sl]) > 1e-9, axis=1)
                entry[f"verbatim_{ch}_pct"] = round(100.0 * float((~touched).mean()), 2)
            out["by_phase"][PHASES[pid] if pid < len(PHASES) else str(pid)] = entry
    return out


def load(path):
    runs = {}
    for p in sorted(Path(path).glob("*_*_*.npz")):
        model, control, seed = p.stem.rsplit("_", 2)
        rec = per_run(p)
        rec["_path"] = str(p)
        runs[(model, control, int(seed))] = rec
    return runs


def merge_summaries(path):
    merged = defaultdict(dict)
    for p in Path(path).glob("summary_*.json"):
        for rec in json.loads(p.read_text()):
            merged[(rec["model"], rec["control"])][rec["seed"]] = rec
    return merged


def table(rows, cols):
    head = "| cell | " + " | ".join(cols) + " |"
    sep = "|---" * (len(cols) + 1) + "|"
    body = []
    for cell, vals in rows.items():
        body.append("| " + " | ".join([f"{cell[0]}/{cell[1]}"] + [vals[c] for c in cols]) + " |")
    print(head)
    print(sep)
    print("\n".join(body))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    args = ap.parse_args()
    base = Path(args.dir)

    runs = load(base)
    if not runs:
        raise SystemExit(f"no runs in {base}")
    summaries = merge_summaries(base)

    cells = defaultdict(list)
    for (model, control, seed), rec in runs.items():
        s = summaries.get((model, control), {}).get(seed, {})
        rec = dict(rec)
        rec["success"] = int(bool(s.get("success", False)))
        rec["cookies"] = int(s.get("cookies_in_target", -1))
        cells[(model, control)].append(rec)

    rows = {}
    for cell, recs in sorted(cells.items()):
        rows[cell] = {
            "n": str(len(recs)),
            "success": f"{sum(r['success'] for r in recs)}/{len(recs)}",
            "cookies_mean": f"{np.mean([r['cookies'] for r in recs]):.1f}",
            "verbatim_%": f"{np.mean([r['verbatim_pct'] for r in recs]):.1f}",
            "rewrite_mm": f"{np.mean([r['rewrite_mm_mean'] for r in recs]):.2f}",
            "model_mm": f"{np.mean([r['model_motion_mm_mean'] for r in recs]):.2f}",
            "max_force_N": f"{np.max([r['max_pad_force'] for r in recs]):.1f}",
            "oracle_reads": f"{int(np.mean([r['oracle_cookie_reads'] + r['oracle_data_reads'] for r in recs]))}",
        }
    print("\n== cells ==")
    table(rows, ["n", "success", "cookies_mean", "verbatim_%", "rewrite_mm", "model_mm",
                 "max_force_N", "oracle_reads"])

    # Validity: with only the safety layer, the executed action must equal the
    # model action except where a hard/soft force or bin-floor rule fired.
    print("\n== authority by channel (percent of steps passed through untouched) ==")
    print("| cell | xyz | rot | grip |")
    print("|---|---|---|---|")
    for cell, recs in sorted(cells.items()):
        print(f"| {cell[0]}/{cell[1]} | "
              f"{np.mean([r['verbatim_xyz_pct'] for r in recs]):.1f} | "
              f"{np.mean([r['verbatim_rot_pct'] for r in recs]):.1f} | "
              f"{np.mean([r['verbatim_grip_pct'] for r in recs]):.1f} |")

    print("\n== self-check: safety control is low-intervention ==")
    for (model, control), recs in sorted(cells.items()):
        if control not in ("safety", "C0"):
            continue
        low = [r for r in recs if r["verbatim_pct"] < 99.0]
        print(f"{model}: min verbatim {min(r['verbatim_pct'] for r in recs):.2f}% "
              f"({len(low)}/{len(recs)} runs below 99%)")

    print("\n== paired raw -> lora (same seeds) ==")
    for control in sorted({c for _, c in cells}):
        deltas, pairs = [], 0
        for (model, ctl, seed), rec in runs.items():
            if ctl != control or model != "lora":
                continue
            base_rec = runs.get(("raw", control, seed))
            if not base_rec:
                continue
            pairs += 1
            deltas.append((rec["verbatim_pct"] - base_rec["verbatim_pct"],
                           rec["rewrite_mm_mean"] - base_rec["rewrite_mm_mean"]))
        if pairs:
            arr = np.asarray(deltas)
            print(f"{control}: n={pairs} d_verbatim={arr[:, 0].mean():+.1f}pp "
                  f"d_rewrite={arr[:, 1].mean():+.2f}mm")

    # A stage releases one channel for one part of the task (C1 only the
    # transport phase), so whole-run numbers cannot separate "the wrapper still
    # owns this channel" from "the model never entered that phase".
    print("\n== authority per phase x channel ==")
    print("| cell | phase | steps | verbatim | xyz | rot | grip | rewrite_mm |")
    print("|---" * 8 + "|")
    for (model, control), recs in sorted(cells.items()):
        names = sorted({n for r in recs for n in r.get("by_phase", {})},
                       key=lambda n: PHASES.index(n) if n in PHASES else len(PHASES))
        for name in names:
            vals = [r["by_phase"][name] for r in recs if name in r.get("by_phase", {})]
            print(f"| {model}/{control} | {name} | {np.mean([v['steps'] for v in vals]):.0f} | "
                  f"{np.mean([v['verbatim_pct'] for v in vals]):.1f} | "
                  f"{np.mean([v['verbatim_xyz_pct'] for v in vals]):.1f} | "
                  f"{np.mean([v['verbatim_rot_pct'] for v in vals]):.1f} | "
                  f"{np.mean([v['verbatim_grip_pct'] for v in vals]):.1f} | "
                  f"{np.mean([v['rewrite_mm_mean'] for v in vals]):.2f} |")

    # The success criterion of the whole ladder: C5 must clearly beat C0 with
    # the same seeds, i.e. the weights -- not the wrapper -- got better. Until a
    # LoRA update lands between stages, C5 is C0 by construction.
    print("\n== C5 vs C0 (same model, same seeds) ==")
    print("criterion: C5 must clearly beat C0; a near-zero delta means the "
          "weights were not updated, not that the model improved")
    for model in sorted({m for m, _ in cells}):
        keys = sorted((k for k in runs if k[0] == model and k[1] == "C5"),
                      key=lambda k: k[2])
        pairs = [(k[2], runs[(model, "C5", k[2])], runs.get((model, "C0", k[2])))
                 for k in keys]
        pairs = [(seed, c5, c0) for seed, c5, c0 in pairs if c0 is not None]
        if not pairs:
            continue
        d = defaultdict(list)
        for seed, c5, c0 in pairs:
            s5 = summaries.get((model, "C5"), {}).get(seed, {})
            s0 = summaries.get((model, "C0"), {}).get(seed, {})
            d["cookies"].append(s5.get("cookies_in_target", -1)
                                - s0.get("cookies_in_target", -1))
            d["success"].append(int(bool(s5.get("success"))) - int(bool(s0.get("success"))))
            for ch in ("xyz", "rot", "grip"):
                d[ch].append(c5[f"verbatim_{ch}_pct"] - c0[f"verbatim_{ch}_pct"])
            d["rewrite"].append(c5["rewrite_mm_mean"] - c0["rewrite_mm_mean"])
        n = len(pairs)
        print(f"{model}: n={n} d_cookies={np.mean(d['cookies']):+.1f} "
              f"d_success={sum(d['success']):+d}/{n} "
              f"d_xyz={np.mean(d['xyz']):+.1f}pp d_rot={np.mean(d['rot']):+.1f}pp "
              f"d_grip={np.mean(d['grip']):+.1f}pp "
              f"d_rewrite={np.mean(d['rewrite']):+.2f}mm")
        # Action-stream check: if the wrapper really released everything, C5 and
        # C0 must be the same trajectory step for step. A difference here is a
        # capability gate that still writes, not a model improvement.
        diffs = []
        for seed, c5, c0 in pairs:
            a = np.load(c5["_path"])["post_14d"]
            b = np.load(c0["_path"])["post_14d"]
            shared = min(len(a), len(b))
            if shared == 0:
                continue
            diff = np.abs(a[:shared] - b[:shared])
            diffs.append((float(diff.max()),
                          int(np.any(diff > 1e-9, axis=1).sum()), shared,
                          abs(len(a) - len(b))))
        if diffs:
            print(f"  trace check: max |C5-C0| {max(x[0] for x in diffs):.2e}, "
                  f"differing steps {sum(x[1] for x in diffs)}/"
                  f"{sum(x[2] for x in diffs)}, length delta "
                  f"{[x[3] for x in diffs]}")

    csv = base / "analysis.csv"
    with csv.open("w") as f:
        cols = ["model", "control", "seed", "success", "cookies", "steps", "verbatim_pct",
                "verbatim_xyz_pct", "verbatim_rot_pct", "verbatim_grip_pct",
                "rewrite_mm_mean", "rewrite_mm_p95", "rewrite_mrad_mean", "model_motion_mm_mean",
                "grip_changed_pct", "max_pad_force", "oracle_cookie_reads", "oracle_data_reads"]
        f.write(",".join(cols) + "\n")
        for (model, control, seed), rec in sorted(runs.items()):
            s = summaries.get((model, control), {}).get(seed, {})
            vals = [model, control, str(seed), str(int(bool(s.get("success", False)))),
                    str(s.get("cookies_in_target", -1))] + [str(rec.get(c, "")) for c in cols[5:]]
            f.write(",".join(vals) + "\n")
    print(f"\n[csv] {csv}")


if __name__ == "__main__":
    main()
