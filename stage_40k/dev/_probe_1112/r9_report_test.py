"""Round 9: exercise the two sections `report40k.py` grew, without a GPU or a checkpoint.

`report40k.py` reads only `out*/step_*/summary.json`, so the whole contract is testable from a
synthetic tree. What has to be proved here, in order of how much it would cost to discover on the
real batch:

* a node whose proprio block is complete renders the band table, the per-column evidence and the
  cost table, and exits 0;
* a node with an *unexplained* off-band reading exits 1 and says so -- this is the case where the
  harness, not the policy, produced the numbers, and the report is the last thing standing between
  it and a published success rate;
* a fixture/test node with no `proprio` and no `timing` block still renders (the flow tests use it)
  and is marked "not recorded" rather than crashing on a missing key;
* the per-column collapse keeps the *widest* episode's raw values, not the first one it saw, and
  marks a column unexplained if any of its episodes is.

Run it wherever it is needed -- it has no pod-only dependency:
    PYTHONDONTWRITEBYTECODE=1 python /workspace/_v1_stage/dev/_probe_1112/r9_report_test.py
"""
from __future__ import annotations

import json
import shutil
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVAL = HERE.parent / "eval40k"
sys.path.insert(0, str(EVAL))

import report40k as R                                              # noqa: E402

BAND_LO = [-0.45, -0.55, 0.70, -1.25, -2.20, -1.90, -0.04, -0.04] + [-3.5] * 6
BAND_HI = [0.30, 0.62, 0.92, 1.05, 2.10, 2.40, 0.04, 0.04] + [9.5] * 6


def exceed(col, seed, nmin, nmax, raw_min, raw_max, fmin, fmax, explained=True):
    w = BAND_HI[col] - BAND_LO[col]
    return {"episode_seed": seed, "column": col, "normalised_min": nmin, "normalised_max": nmax,
            "band_lo": BAND_LO[col], "band_hi": BAND_HI[col], "raw_min": raw_min,
            "raw_min_frame": fmin, "raw_max": raw_max, "raw_max_frame": fmax,
            "raw_min_below_band_lo_by_widths": round((BAND_LO[col] - raw_min) / w, 3),
            "raw_max_above_band_hi_by_widths": round((raw_max - BAND_HI[col]) / w, 3),
            "explained_by_raw_excursion": explained}


def force_block():
    ch = ["fx", "fy", "fz", "tx", "ty", "tz"]
    return {"channel_names": ch, "coverage": 0.81, "overall": {"mae": [0.4] * 6},
            "in_contact": {"mae": [0.6] * 6},
            "baselines": {"persistence": {"mae": [0.9] * 6}, "zero": {"mae": [1.5] * 6}},
            "mae_vs_persistence": 0.5, "mae_vs_zero": 0.3, "by_horizon": {}}


def records():
    out = []
    for i, seed in enumerate([11, 12, 13]):
        out.append({"seed": seed, "frames": 601, "success": seed == 13, "nan_abort_frame": -1,
                    "wall_s": 210.0 + i, "sim_s_total": 110.0 + i, "infer_s_total": 100.0 + i,
                    "sim_s_max_step": 0.75 if seed != 12 else 10.93, "infer_s_share": 0.47,
                    "unaccounted_s": 0.2, "cpu_throttled_usec_delta": 0})
    return out


def summary(step=40000, off_band=True, unexplained=False, witness_gap=False, band_diff=False,
            backend="env", policy="merged"):
    seeds = [11, 12, 13]
    doc = {"step": step, "status": "ran", "backend": backend, "policy": policy, "seeds": seeds,
           "episodes": len(seeds), "successes": 1, "success_rate": 0.3333,
           "success_ci95": [0.06, 0.79], "mean_frames": 601.0, "nan_aborts": 0, "wall_s": 630.0,
           "force": force_block(), "records": records(),
           "calibration": {"channels": [0, 1, 2, 12, 13, 16]}, "force_contract_channels": [0, 1, 2,
                                                                                           12, 13,
                                                                                           16],
           "light_node": "n/a", "scene": {"position_noise_m": 0.0, "yaw_noise_rad": 0.0,
                                          "randomize_boxes": False, "distinct_scene_keys": 1,
                                          "episodes": len(seeds)},
           "eval_settings": {}, "fw_import": {}}
    if backend == "fixture":
        doc["fixture"] = {"force_mode": "const", "stub_offset": 0.0, "frames_cap": 40}
        doc["expected_mae"] = 0.0
        return doc
    ex = []
    if off_band:
        ex = [exceed(2, 11, -5.0, 5.0, 0.782, 1.8027, 50, 590),
              exceed(2, 12, -5.0, 5.0, 0.782, 2.10, 50, 40),        # wider: must win the collapse
              exceed(9, 12, -5.0, 5.0, -0.93, 372.97, 70, 580)]
        if unexplained:
            ex.append(exceed(0, 13, -5.0, 0.21, -0.12, 0.21, 590, 210, explained=False))
    doc["proprio"] = {
        "band": {"source": "a3_fw_v1g/dataset_stats.json", "lo": BAND_LO, "hi": BAND_HI},
        "clamped_calls": 56 if off_band else 0, "clamped_columns": [0, 2, 8, 9] if off_band else [],
        "exceedances": ex, "unexplained": 1 if unexplained else 0,
        "band_map_worst_diff": (0.5 if band_diff else 4.5e-07),
        "episodes_paired": len(seeds), "episodes": len(seeds),
        "episodes_without_witness": [13] if witness_gap else []}
    walls = [float(r["wall_s"]) for r in doc["records"]]
    sims = [float(r["sim_s_total"]) for r in doc["records"]]
    infers = [float(r["infer_s_total"]) for r in doc["records"]]

    def stat(xs):
        return {"mean": round(statistics.fmean(xs), 1), "median": round(statistics.median(xs), 1),
                "min": round(min(xs), 1), "max": round(max(xs), 1)}
    doc["timing"] = {
        "per_episode_s": [{"seed": r["seed"], "wall": r["wall_s"], "sim": r["sim_s_total"],
                           "infer": r["infer_s_total"], "unaccounted": r["unaccounted_s"],
                           "cpu_throttled_s": 0.0} for r in doc["records"]],
        "wall": stat(walls), "sim": stat(sims), "infer": stat(infers),
        "cpu_throttled_s_total": 0.0,
        "projected_160_episodes_h_median": round(statistics.median(walls) * 160 / 3600, 2),
        "projected_160_episodes_h_mean": round(statistics.fmean(walls) * 160 / 3600, 2),
        "this_run_h": round(sum(walls) / 3600, 3)}
    return doc


def write_tree(root, docs):
    for d in docs:
        p = Path(root) / ("step_%06d" % d["step"])
        p.mkdir(parents=True, exist_ok=True)
        (p / "summary.json").write_text(json.dumps(d, indent=2, sort_keys=True, default=str))


def run(out_dir, steps=(40000,)):
    md = Path(out_dir) / "report.md"
    js = Path(out_dir) / "report.json"
    cmd = [sys.executable, str(EVAL / "report40k.py"), "--out", str(out_dir),
           "--steps"] + [str(s) for s in steps] + ["--md", str(md), "--json", str(js)]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(EVAL))
    return {"rc": res.returncode, "md": md.read_text() if md.is_file() else "",
            "json": json.loads(js.read_text()) if js.is_file() else {},
            "stderr": res.stderr, "stdout_last": res.stdout.strip().splitlines()[-1:]}


checks, fails = [], []


def check(name, ok, detail=""):
    checks.append(name)
    if not ok:
        fails.append("%s %s" % (name, detail))
    print("%-58s %s %s" % (name, "PASS" if ok else "FAIL", detail if not ok else ""), flush=True)


def main():
    tmp = Path(tempfile.mkdtemp(prefix="r9_report_"))
    try:
        # 1. the clean case: band + timing recorded, nothing unexplained
        write_tree(tmp / "clean", [summary()])
        r = run(tmp / "clean")
        check("clean render exits 0", r["rc"] == 0, r["stderr"][-400:])
        check("clean: band section present",
              "proprioception against the training band" in r["md"])
        check("clean: cost section present", "What the run cost" in r["md"])
        check("clean: unexplained sub-table says zero", "| 40000 | 0 |" in r["md"])
        check("clean: no harness indictment", "indicts the harness" not in r["md"])
        check("clean: off-band warning is stated as a policy result",
              "left the recorded training band" in r["md"] and "not a harness defect" in r["md"])
        check("clean: per-column collapse keeps the widest episode",
              "| 2 | 0.7000 | 0.9200 | 0.7820 (50) | 2.1000 (40) |" in r["md"],
              [ln for ln in r["md"].splitlines() if ln.startswith("| 40000 | 2 |")])
        check("clean: exit JSON reports the count",
              json.loads(r["stdout_last"][0])["unexplained_proprio"] == 0, r["stdout_last"])
        check("clean: projection range line", "the 160-episode projection from them is" in r["md"])
        check("clean: json carries both tables",
              len(r["json"]["band_table"]) == 1 and len(r["json"]["timing_table"]) == 1)
        check("clean: worst single MuJoCo step from records",
              r["json"]["timing_table"][0]["worst_single_sim_step_s"] == 10.93)

        # 2. the case that must stop the batch: an unexplained off-band reading
        write_tree(tmp / "forged", [summary(unexplained=True)])
        f = run(tmp / "forged")
        check("unexplained: exit code is 1", f["rc"] == 1, "rc=%s" % f["rc"])
        check("unexplained: names the column", "columns [0]" in f["md"])
        check("unexplained: indicts the harness", "indicts the harness" in f["md"])
        check("unexplained: per-column table marks the column NO",
              any(ln.startswith("| 40000 | 0 |") and ln.rstrip().endswith("| NO |")
                  for ln in f["md"].splitlines()))

        # 3. fixture/test node: no proprio block, no timing block
        fx = summary(backend="fixture", policy="stub")
        write_tree(tmp / "fx", [fx])
        t = run(tmp / "fx")
        check("fixture: renders without the blocks", t["rc"] == 0, t["stderr"][-400:])
        check("fixture: band row says not recorded",
              "| 40000 **(test run)** | - | not recorded |" in t["md"])
        check("fixture: cost row says not measured", "not measured" in t["md"])
        check("fixture: no missing-band warning", "has no `proprio` block" not in t["md"])
        check("fixture: expected-MAE verdict still rendered", "flow test: expected MAE" in t["md"])

        # 4. a real node whose summary is missing the evidence entirely
        bare = summary()
        del bare["proprio"], bare["timing"]
        write_tree(tmp / "bare", [bare])
        b = run(tmp / "bare")
        check("no-evidence node: still renders", b["rc"] == 0, b["stderr"][-400:])
        check("no-evidence node: warns the evidence is missing",
              "has no `proprio` block" in b["md"] and "has no `timing` block" in b["md"])

        # 5. witness gap and a band map that disagrees
        write_tree(tmp / "gap", [summary(witness_gap=True)])
        g = run(tmp / "gap")
        check("witness gap: warned and listed",
              "no per-replan raw proprio" in g["md"] and "seeds [13]" in g["md"])
        write_tree(tmp / "diff", [summary(band_diff=True)])
        d = run(tmp / "diff")
        check("band-map drift: warned above tolerance",
              "more than a float32 round-trip" in d["md"])

        # 6. the collapse itself, on the function rather than through markdown
        per = R._column_worsts([exceed(2, 11, -5.0, 5.0, 0.782, 1.10, 50, 100),
                                exceed(2, 12, -5.0, 5.0, 0.782, 2.10, 50, 40),
                                exceed(2, 13, -5.0, 5.0, 0.782, 1.20, 50, 30)])
        check("collapse: one row per column", list(per) == [2], list(per))
        check("collapse: keeps the farthest raw max", per[2]["raw_max"] == 2.10, per[2])
        check("collapse: counts every seed that left the band", per[2]["episodes"] == 3, per[2])
        mixed = R._column_worsts([exceed(2, 11, -5.0, 5.0, 0.782, 2.10, 50, 40),
                                  exceed(2, 12, -5.0, 5.0, 0.782, 1.10, 50, 40, explained=False)])
        check("collapse: one unexplained episode marks the column unexplained",
              mixed[2]["explained"] is False)
        check("collapse: the widest episode still wins when it is the explained one",
              mixed[2]["raw_max"] == 2.10 and mixed[2]["worst_episode_seed"] == 11, mixed[2])
        # the same three episodes in the order where the widest arrives first: the row that won
        # must not reset the seed list on the two that follow it
        first = R._column_worsts([exceed(2, 12, -5.0, 5.0, 0.782, 2.10, 50, 40),
                                  exceed(2, 11, -5.0, 5.0, 0.782, 1.10, 50, 100),
                                  exceed(2, 13, -5.0, 5.0, 0.782, 1.20, 50, 30)])
        check("collapse: order does not change which seeds are counted",
              first[2]["seeds"] == [11, 12, 13] and first[2]["episodes"] == 3
              and first[2]["worst_episode_seed"] == 12, first[2])
        check("collapse: same rows in and out of order",
              {k: v for k, v in first[2].items() if k != "worst_episode_seed"}
              == {k: v for k, v in per[2].items() if k != "worst_episode_seed"},
              (first[2], per[2]))
        check("collapse: an empty exceedance list collapses to nothing",
              R._column_worsts([]) == {} and R._column_worsts(None) == {})

        # 7. the constant this file duplicates on purpose must match the driver it mirrors
        src = (EVAL / "eval40k.py").read_text(errors="replace")
        line = [ln for ln in src.splitlines() if ln.startswith("PROPRIO_BASE_DIM =")]
        check("PROPRIO_BASE_DIM matches eval40k.py", len(line) == 1
              and int(line[0].split("=")[1].split("#")[0]) == R.PROPRIO_BASE_DIM, line)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(json.dumps({"n_checks": len(checks), "n_failed": len(fails), "failed": fails}))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
