"""Render the eight nodes side by side, from the summaries the driver already wrote.

Separate from `eval40k.py` because reading a report should never require GPU, a checkpoint
or a simulator: this file only opens `out*/step_*/summary.json`. It exists to answer the
question the brief asks -- "can all eight nodes be recovered, deployed and evaluated
independently, and how do they compare" -- in one place, with the caveats attached to the
numbers instead of underneath them:

* the seed list is identical for every node, and every seed gets exactly one episode, so
  successes are compared *paired* per seed. Twenty episodes cannot resolve a few percent of
  difference -- at 10/20 the Wilson interval is [0.30, 0.70] -- so the paired counts come with
  the honest note that they are a direction, not a result;
* a run whose backend is `fixture` or whose policy is `stub` is labelled as a test, because
  those numbers describe the fixture's seed rule and would otherwise read as model results;
* a force MAE without a reference is meaningless, so each node prints its own persistence and
  zero baselines from the same scored steps and the ratio alongside;
* a normalised proprio value outside [-1, 1] has two possible causes -- the robot left the
  42,681 training rows, or this harness normalised the wrong numbers -- so each node prints the
  raw value and frame that put it there, and separately the count of excursions *nothing*
  explains, which has to be zero for any of the other columns to mean anything;
* the batch is ~160 episodes of MuJoCo and the decision to run it is made on that number, so the
  simulation/inference split each node measured is printed with a projection from its median.

    cd /workspace/_v1_stage/dev/eval40k
    PYTHONDONTWRITEBYTECODE=1 python report40k.py --out out
    PYTHONDONTWRITEBYTECODE=1 python report40k.py --out out --md out/report.md

The last line of stdout is a one-object JSON summary; the exit code is 1 if any node reported a
proprio reading that left the training band with no raw excursion behind it, because in that case
the harness -- not the policy -- is what produced the numbers above it.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

NODE_INTERVAL = 5000
TOTAL_STEPS = 40000
TEST_LABELS = ("fixture", "stub")
# mirrors eval40k.PROPRIO_BASE_DIM: the first columns are the worker's own 8 (EEF pose 3,
# axis-angle 3, 2 finger joints) and the last 6 are the tared force readings. This file must not
# import eval40k to learn it -- importing that pulls torch, and reading a report should never
# need a GPU -- so `flow_test.py` compares the two numbers at the source level instead.
PROPRIO_BASE_DIM = 8
BAND_MAP_TOLERANCE = 1e-4      # recomputed band vs the worker's; float32 round-trip is ~1e-7


def node_steps():
    return [NODE_INTERVAL * i for i in range(1, TOTAL_STEPS // NODE_INTERVAL + 1)]


def load_summaries(out_dir):
    out_dir = Path(out_dir)
    found = {}
    for path in sorted(out_dir.glob("step_*/summary.json")):
        try:
            doc = json.loads(path.read_text())
        except Exception as e:                                  # noqa: BLE001
            print(f"WARN unreadable {path}: {e}", flush=True)
            continue
        step = int(doc.get("step") or path.parent.name.split("_")[-1])
        found[step] = doc
    return found


def is_test(doc):
    return doc.get("backend") in TEST_LABELS or doc.get("policy") in TEST_LABELS


def paired(a, b):
    """Per-seed wins/losses/ties of node `b` against node `a` over their shared seeds."""
    sa = {int(r["seed"]): bool(r["success"]) for r in a.get("records", [])}
    sb = {int(r["seed"]): bool(r["success"]) for r in b.get("records", [])}
    shared = sorted(set(sa) & set(sb))
    wins = [s for s in shared if sb[s] and not sa[s]]
    losses = [s for s in shared if sa[s] and not sb[s]]
    ties = [s for s in shared if sa[s] == sb[s]]
    return {"reference_step": a.get("step"), "shared_seeds": len(shared),
            "wins": len(wins), "losses": len(losses), "ties": len(ties),
            "win_seeds": wins, "loss_seeds": losses,
            "identical_seed_list": sorted(sa) == sorted(sb)}


def frame_delta(a, b):
    sa = {int(r["seed"]): int(r["frames"]) for r in a.get("records", [])}
    sb = {int(r["seed"]): int(r["frames"]) for r in b.get("records", [])}
    shared = sorted(set(sa) & set(sb))
    d = [sb[s] - sa[s] for s in shared]
    return {"mean_frames_delta": (round(float(np.mean(d)), 1) if d else None),
            "median_frames_delta": (round(float(np.median(d)), 1) if d else None),
            "note": "negative means the later node finished the same seed faster"}


def fmt_ci(doc):
    rate, ci = doc.get("success_rate"), doc.get("success_ci95")
    if rate is None or not ci:
        return "-"
    return "%.3f [%.3f, %.3f] (%d/%d)" % (rate, ci[0], ci[1],
                                          doc.get("successes", 0), doc.get("episodes", 0))


def force_line(doc):
    f = doc.get("force") or {}
    ov = (f.get("overall") or {}).get("mae")
    if not ov:
        return "no scored force steps"
    return "mean %.3f N, worst %.3f N (%s)" % (
        float(np.mean(ov)), float(np.max(ov)), f.get("channel_names", [])[int(np.argmax(ov))])


def table_rows(steps, docs):
    rows = []
    for s in steps:
        doc = docs.get(s)
        if doc is None:
            rows.append({"step": s, "status": "not evaluated"})
            continue
        r = {"step": s, "status": doc.get("status"),
             "backend": doc.get("backend"), "policy": doc.get("policy"),
             "test_only": is_test(doc), "episodes": doc.get("episodes"),
             "success": fmt_ci(doc), "mean_frames": doc.get("mean_frames"),
             "nan_aborts": doc.get("nan_aborts"), "wall_s": doc.get("wall_s"),
             "force": force_line(doc),
             "coverage": (doc.get("force") or {}).get("coverage"),
             "mae_vs_persistence": (doc.get("force") or {}).get("mae_vs_persistence"),
             "mae_vs_zero": (doc.get("force") or {}).get("mae_vs_zero"),
             "seeds": doc.get("seeds")}
        prev = docs.get(s - NODE_INTERVAL)
        if prev is not None:
            r["paired_vs_prev"] = paired(prev, doc)
            r["frames_vs_prev"] = frame_delta(prev, doc)
        rows.append(r)
    return rows


def channel_table(steps, docs):
    names = None
    for doc in docs.values():
        if doc.get("force", {}).get("channel_names"):
            names = doc["force"]["channel_names"]
            break
    if not names:
        return {}
    out = {"channels": names, "overall_mae_N": {}, "in_contact_mae_N": {},
           "persistence_mae_N": {}, "zero_mae_N": {}, "by_horizon_mean_mae_N": {}}
    for s in steps:
        f = (docs.get(s) or {}).get("force")
        if not f or not f.get("overall", {}).get("mae"):
            continue
        out["overall_mae_N"][s] = f["overall"]["mae"]
        out["in_contact_mae_N"][s] = (f.get("in_contact") or {}).get("mae")
        out["persistence_mae_N"][s] = (f["baselines"]["persistence"] or {}).get("mae")
        out["zero_mae_N"][s] = (f["baselines"]["zero"] or {}).get("mae")
        out["by_horizon_mean_mae_N"][s] = {
            k: (round(float(np.mean(v["mae"])), 4) if v.get("mae") else None)
            for k, v in (f.get("by_horizon") or {}).items()}
    return out


def _column_worsts(exceedances):
    """One row per proprio column that left the recorded band, keeping the farthest episode.

    `eval40k.py` writes an exceedance per (episode, replan extreme, column); across twenty
    episodes that is more lines than anyone reads. What the reader needs per column is how far
    past the recorded edge the raw value went, in band widths, which episode did it, and whether
    the excursion actually accounts for the normalised number -- so this collapses on the widest
    excursion and remembers every seed that contributed.
    """
    per = {}
    for e in exceedances or []:
        c = int(e["column"])
        widths = max(0.0, float(e.get("raw_min_below_band_lo_by_widths") or 0.0),
                     float(e.get("raw_max_above_band_hi_by_widths") or 0.0))
        cur = per.get(c)
        if cur is None:
            cur = per[c] = {"widths_past_edge": -1.0, "seeds": set(), "explained": True}
        # the winner's fields replace the loser's, but the seed list and the explained flag are
        # column properties: overwriting the dict here is how a column silently stops reporting
        # the episodes that left the band less far
        if widths > cur["widths_past_edge"]:
            cur.update({"widths_past_edge": round(widths, 3),
                        "band_lo": e.get("band_lo"), "band_hi": e.get("band_hi"),
                        "normalised_min": e.get("normalised_min"),
                        "normalised_max": e.get("normalised_max"),
                        "raw_min": e.get("raw_min"), "raw_min_frame": e.get("raw_min_frame"),
                        "raw_max": e.get("raw_max"), "raw_max_frame": e.get("raw_max_frame"),
                        "worst_episode_seed": e.get("episode_seed")})
        cur["explained"] = cur["explained"] and bool(e.get("explained_by_raw_excursion"))
        if e.get("episode_seed") is not None:
            cur["seeds"].add(int(e["episode_seed"]))
    for col in per.values():
        col["seeds"] = sorted(col["seeds"])
        col["episodes"] = len(col["seeds"])
    return per


def band_rows(steps, docs):
    """The per-node proprioception evidence: what left the band, and whether anything is unexplained.

    `unexplained` is the only column here that can indict this harness. A normalised reading of
    +-5 is the clamp, and the clamp only reaches +-5 at two band widths past the recorded edge,
    so a value that big demands an excursion that far in the raw column of the same replan;
    `eval40k.py` counts the cases where it does not find one. Every other column here is a
    statement about where the robot went.
    """
    rows = []
    for s in steps:
        doc = docs.get(s)
        if doc is None:
            continue
        p = doc.get("proprio")
        if not p:
            rows.append({"step": s, "recorded": False, "test_only": is_test(doc)})
            continue
        per = _column_worsts(p.get("exceedances"))
        base = sorted(c for c in per if c < PROPRIO_BASE_DIM)
        force = sorted(c for c in per if c >= PROPRIO_BASE_DIM)
        rows.append({"step": s, "recorded": True, "test_only": is_test(doc),
                     "episodes": p.get("episodes"),
                     "episodes_off_band": len({e.get("episode_seed")
                                               for e in (p.get("exceedances") or [])}),
                     "base_columns_off_band": base, "force_columns_off_band": force,
                     "worst_base_band_widths": (max([per[c]["widths_past_edge"] for c in base])
                                                if base else None),
                     "worst_force_band_widths": (max([per[c]["widths_past_edge"] for c in force])
                                                 if force else None),
                     "clamped_calls": p.get("clamped_calls"),
                     "clamped_columns": p.get("clamped_columns"),
                     "unexplained": p.get("unexplained"),
                     "band_map_worst_diff": p.get("band_map_worst_diff"),
                     "episodes_paired": p.get("episodes_paired"),
                     "episodes_without_witness": p.get("episodes_without_witness"),
                     "columns": {str(c): per[c] for c in sorted(per)},
                     "band_source": (p.get("band") or {}).get("source")})
    return rows


def timing_rows(steps, docs):
    """What each node's episodes actually cost, and the 160-episode number they project to.

    The projection is the batch driver's own field, made from the median wall time per episode;
    the mean sits next to it because one CPU-starved episode inflates it without making any other
    episode slower, and the two disagreed by 15x on the run that produced the first estimate.
    """
    rows = []
    for s in steps:
        doc = docs.get(s)
        if doc is None:
            continue
        t = doc.get("timing")
        if not t:
            rows.append({"step": s, "recorded": False, "test_only": is_test(doc or {})})
            continue
        steps_max = [float(r.get("sim_s_max_step") or 0.0) for r in (doc.get("records") or [])]
        rows.append({"step": s, "recorded": True, "test_only": is_test(doc),
                     "episodes": doc.get("episodes"), "wall": t.get("wall"),
                     "sim": t.get("sim"), "infer": t.get("infer"),
                     "infer_share_mean": (round(float(np.mean([float(r.get("infer_s_share") or 0.0)
                                                              for r in doc.get("records") or []])), 3)
                                          if doc.get("records") else None),
                     "worst_single_sim_step_s": round(max(steps_max), 3) if steps_max else None,
                     "unaccounted_s_max": (max([float(e.get("unaccounted") or 0.0)
                                                for e in t.get("per_episode_s") or []], default=None)),
                     "cpu_throttled_s_total": t.get("cpu_throttled_s_total"),
                     "this_run_h": t.get("this_run_h"),
                     "projected_160_h_median": t.get("projected_160_episodes_h_median"),
                     "projected_160_h_mean": t.get("projected_160_episodes_h_mean"),
                     "per_episode": t.get("per_episode_s")})
    return rows


def _n(v, fmt="%.3f"):
    return "-" if v is None else (fmt % v if isinstance(v, (int, float)) else str(v))


def _stat(v, key="median", fmt="%.1f"):
    return "-" if not v else _n(v.get(key), fmt)


def to_markdown(doc):
    lines = ["# 40k force-conditioned checkpoints: evaluation report", "",
             "generated: %s" % doc["generated"],
             "seed list (same for every node): %s" % doc["seed_list"], ""]
    if doc.get("warnings"):
        lines.append("## Warnings")
        lines += ["- %s" % w for w in doc["warnings"]]
        lines.append("")
    lines += ["## Node comparison", "",
              "| step | status | episodes | success rate | mean frames | force MAE | "
              "coverage | vs persistence | vs zero | paired vs prev (W/L/T) |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in doc["rows"]:
        if r.get("status") == "not evaluated":
            lines.append("| %d | not evaluated | - | - | - | - | - | - | - | - |" % r["step"])
            continue
        p = r.get("paired_vs_prev")
        pair = "-" if not p else "%d/%d/%d" % (p["wins"], p["losses"], p["ties"])
        flag = " **(test run)**" if r.get("test_only") else ""
        lines.append("| %d%s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            r["step"], flag, r["status"], r["episodes"], r["success"], r["mean_frames"],
            r["force"], r["coverage"], r["mae_vs_persistence"], r["mae_vs_zero"], pair))
    lines.append("")
    ct = doc["channel_table"]
    if ct:
        lines += ["## Force error per channel (newtons)", "",
                  "| step | " + " | ".join(ct["channels"]) + " |",
                  "| --- | " + " | ".join(["---"] * len(ct["channels"])) + " |"]
        for s, mae in sorted(ct["overall_mae_N"].items()):
            lines.append("| %d | " % s + " | ".join("%.3f" % v for v in mae) + " |")
        lines.append("")
        lines += ["### The same nodes against the two dumb predictors", "",
                  "| step | model mean | persistence mean | zero mean |", "| --- | --- | --- | --- |"]
        for s in sorted(ct["overall_mae_N"]):
            m = lambda d: float(np.mean(d[s])) if d.get(s) else None  # noqa: E731
            pm, zm = m(ct["persistence_mae_N"]), m(ct["zero_mae_N"])
            mm = m(ct["overall_mae_N"])
            lines.append("| %d | %s | %s | %s |" % (
                s, "n/a" if mm is None else "%.3f" % mm,
                "n/a" if pm is None else "%.3f" % pm,
                "n/a" if zm is None else "%.3f" % zm))
        lines.append("")
    bt = doc.get("band_table") or []
    if bt:
        lines += ["## What the model was handed: proprioception against the training band", "",
                  "Each normalised column should sit in [-1, 1] if the observation is inside the "
                  "range the 42,681 training rows were measured over; the harness clamps at +-5, "
                  "which is two band widths past a recorded edge. `unexplained` counts the values "
                  "for which the raw column of the same replan does not show an excursion that "
                  "large -- that is the only column here that can indict the harness, and every "
                  "other one is a statement about where the robot went.", "",
                  "| step | episodes | episodes off band | base cols off band | worst base "
                  "excursion (widths) | force cols off band | worst force excursion (widths) | "
                  "clamped replans | band-map worst diff | episodes without witness |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for r in bt:
            if not r.get("recorded"):
                lines.append("| %d%s | - | not recorded | - | - | - | - | - | - | - |"
                             % (r["step"], " **(test run)**" if r.get("test_only") else ""))
                continue
            flag = " **(test run)**" if r.get("test_only") else ""
            lines.append("| %d%s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                r["step"], flag, _n(r["episodes"], "%d"), _n(r["episodes_off_band"], "%d"),
                ",".join(str(c) for c in r["base_columns_off_band"]) or "none",
                _n(r["worst_base_band_widths"]),
                ",".join(str(c) for c in r["force_columns_off_band"]) or "none",
                _n(r["worst_force_band_widths"]), _n(r["clamped_calls"], "%d"),
                "%.1e" % r["band_map_worst_diff"] if r["band_map_worst_diff"] is not None else "-",
                ",".join(str(s) for s in r["episodes_without_witness"]) or "none"))
        lines.append("")
        lines += ["### `unexplained` per node (must be zero)", "",
                  "| step | unexplained | note |", "| --- | --- | --- |"]
        for r in bt:
            if not r.get("recorded"):
                continue
            n_un = r["unexplained"]
            note = ("every off-band reading is attributed to a raw excursion at least that far, "
                    "in the same replan" if not n_un else
                    "LOOK HERE FIRST: %s value(s) left the band with no raw excursion behind them"
                    % n_un)
            lines.append("| %d | %s | %s |" % (r["step"], _n(n_un, "%d"), note))
        lines.append("")
        detail = [r for r in bt if r.get("recorded") and r.get("columns")]
        if detail:
            lines += ["### Per-column evidence, worst episode of each node", "",
                      "Columns 0-7 are the worker's own proprio (EEF x/y/z, roll/pitch/yaw as "
                      "axis-angle, 2 finger joints); 8-13 are the six zero-load-tared force "
                      "readings. `widths past edge` is how far the raw extreme sits past the "
                      "nearer recorded edge, in units of that column's own band width, measured on "
                      "the frame named.", "",
                      "| step | col | band lo | band hi | raw min (frame) | raw max (frame) | "
                      "normalised range | widths past edge | episodes | explained |",
                      "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
            for r in detail:
                for col, c in sorted(r["columns"].items(), key=lambda kv: int(kv[0])):
                    lines.append("| %d | %s | %s | %s | %s (%s) | %s (%s) | [%s, %s] | %s | %d | %s |"
                                 % (r["step"], col, _n(c["band_lo"], "%.4f"),
                                    _n(c["band_hi"], "%.4f"),
                                    _n(c["raw_min"], "%.4f"), _n(c["raw_min_frame"], "%d"),
                                    _n(c["raw_max"], "%.4f"), _n(c["raw_max_frame"], "%d"),
                                    _n(c["normalised_min"], "%.2f"),
                                    _n(c["normalised_max"], "%.2f"),
                                    _n(c["widths_past_edge"]), c["episodes"],
                                    "yes" if c["explained"] else "NO"))
            lines.append("")
    tt = doc.get("timing_table") or []
    if tt:
        lines += ["## What the run cost, measured on the machine it ran on", "",
                  "The batch decision is made from these numbers, so they come from the same "
                  "summaries as the success rates rather than from an expectation. Projection is "
                  "the median wall time per episode times 160; the mean is next to it because one "
                  "CPU-starved episode inflates the mean without making any other episode faster.",
                  "",
                  "| step | episodes | wall median s | wall mean s | sim median s | infer median s "
                  "| infer share | worst single MuJoCo step s | cpu throttled s | this run h | "
                  "projected 160 episodes h (median / mean) |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for r in tt:
            if not r.get("recorded"):
                lines.append("| %d%s | %s | not measured | - | - | - | - | - | - | - | - |"
                             % (r["step"], " **(test run)**" if r.get("test_only") else "",
                                _n(r.get("episodes"), "%d")))
                continue
            flag = " **(test run)**" if r.get("test_only") else ""
            lines.append("| %d%s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s / %s |" % (
                r["step"], flag, _n(r["episodes"], "%d"), _stat(r["wall"]),
                _n((r["wall"] or {}).get("mean"), "%.1f"), _stat(r["sim"]), _stat(r["infer"]),
                _n(r["infer_share_mean"]), _n(r["worst_single_sim_step_s"], "%.3f"),
                _n(r["cpu_throttled_s_total"], "%.1f"), _n(r["this_run_h"], "%.3f"),
                _n(r["projected_160_h_median"], "%.2f"),
                _n(r["projected_160_h_mean"], "%.2f")))
        lines.append("")
        proj = [r["projected_160_h_median"] for r in tt
                if r.get("recorded") and r.get("projected_160_h_median") is not None]
        if proj:
            span = ("%.2f h" % proj[0] if len(set(proj)) == 1
                    else "%.2f-%.2f h" % (min(proj), max(proj)))
            lines.append("%d node(s) measured their own cost; the 160-episode projection from "
                         "them is %s, and because the batch loads one 11 GiB artifact at a time "
                         "the wall-clock total is the sum over nodes, not the fastest of them. "
                         "Treat the spread as the uncertainty in the estimate, not as a choice of "
                         "node." % (len(proj), span))
            lines.append("")
    lines += ["## How to read this",
              "- Success rate is from the environment's own `info` verdict, Wilson 95% CI; "
              "20 episodes per node, one per screened seed and no repeats, so the interval is "
              "wider than most real differences.",
              "- `paired vs prev` counts seeds where the later node succeeded and the earlier "
              "one failed (and the reverse). It is the only comparison that controls for "
              "seed difficulty, and at this sample size it says a direction, not a result.",
              "- Force MAE is scored only on futures that actually happened. Coverage below 1 "
              "is expected: the last chunk of every episode predicts frames the episode never "
              "reached, and imputing them would reward predicting nothing.",
              "- `vs persistence`/`vs zero` are ratios of the model's MAE to a constant-force "
              "and a no-contact predictor on the same steps. A value >= 1 means the head "
              "carries no information about the future beyond the current reading.",
              "- Runs labelled **(test run)** come from the fixture backend or the stub policy; "
              "they validate the pipeline and describe no policy.",
              "- `seed list` is not `scene list`. With the randomisation knobs at zero the "
              "cookies start in exactly the same place in every episode and the seed only "
              "re-draws the policy's diffusion noise, so the episode count buys repeatability, "
              "not layout coverage. Each node's `scene` block says which of the two it is.",
              "- An off-band proprio column is a policy result, not a data-quality complaint: it "
              "means that episode's arm reached a pose or a force the recorded rows never "
              "contained, which is exactly where an extrapolating diffusion policy has to be "
              "believed least. The ±1 values themselves are not corrupt and the training band is "
              "not corrected for them.",
              "- The cost table describes the machine as much as the model. If `worst single "
              "MuJoCo step` is seconds rather than milliseconds, the run was simulator-bound and "
              "the GPU was waiting; that is the number the batch estimate should be read against.", ""]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "out"))
    ap.add_argument("--steps", type=int, nargs="+", default=node_steps())
    ap.add_argument("--md", default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    docs = load_summaries(args.out)
    band = band_rows(args.steps, docs)
    timing = timing_rows(args.steps, docs)
    warnings = []
    seed_lists = {tuple(d.get("seeds") or []) for d in docs.values() if d.get("seeds")}
    if len(seed_lists) > 1:
        warnings.append("nodes were run with different seed lists, so the paired columns "
                        "only cover shared seeds: %s" % [list(s) for s in seed_lists])
    tests = [s for s, d in docs.items() if is_test(d)]
    if tests:
        warnings.append("these steps are test runs (fixture backend or stub policy), not "
                        "model results: %s" % sorted(tests))
    for s, d in sorted(docs.items()):
        sc = d.get("scene") or {}
        if sc.get("distinct_scene_keys") == 1 and int(sc.get("episodes") or 0) > 1:
            warnings.append("step %d: all %d episodes start from the same scene "
                            "(position_noise_m=%s, yaw_noise_rad=%s, randomize_boxes=%s), so "
                            "this success rate is repeatability on one layout, not "
                            "generalisation across layouts"
                            % (s, sc["episodes"], sc.get("position_noise_m"),
                               sc.get("yaw_noise_rad"), sc.get("randomize_boxes")))
    missing = [s for s in args.steps if s not in docs]
    if missing:
        warnings.append("not evaluated yet: %s" % missing)
    for r in band:
        if not r.get("recorded"):
            if not r.get("test_only"):
                warnings.append("step %d: the summary has no `proprio` block, so nothing in this "
                                "report can tell a harness normalisation bug apart from the robot "
                                "leaving the training band for this node" % r["step"])
            continue
        if r.get("unexplained"):
            bad = sorted(int(c) for c, v in r["columns"].items() if not v["explained"])
            warnings.append("step %d: %d proprio reading(s) handed to the model left the recorded "
                            "band with no raw excursion in the same replan behind them (columns "
                            "%s) -- that indicts the harness, not the policy, and the batch should "
                            "stop on it rather than report a success rate"
                            % (r["step"], r["unexplained"], bad))
        if r.get("episodes_without_witness"):
            warnings.append("step %d: %d episode(s) recorded no per-replan raw proprio, so any "
                            "band excursion they show is unattributable: seeds %s"
                            % (r["step"], len(r["episodes_without_witness"]),
                               r["episodes_without_witness"]))
        if r.get("episodes_paired") != r.get("episodes"):
            warnings.append("step %d: the per-replan band map paired on %s of %s episodes, so the "
                            "attribution below covers fewer episodes than the success rate"
                            % (r["step"], r.get("episodes_paired"), r.get("episodes")))
        diff = r.get("band_map_worst_diff")
        if diff is not None and diff > BAND_MAP_TOLERANCE:
            warnings.append("step %d: recomputing the normalisation from the worker's own band "
                            "disagrees with what the model was handed by %.2e, which is more than "
                            "a float32 round-trip" % (r["step"], diff))
        if r.get("episodes_off_band"):
            worst = max([w for w in (r.get("worst_base_band_widths"),
                                     r.get("worst_force_band_widths")) if w is not None],
                        default=None)
            warnings.append("step %d: %d of %d episodes left the recorded training band "
                            "(base columns %s, force columns %s, farthest %s band widths past a "
                            "recorded edge; the +-5 clamp engaged on %s replans) -- every one of "
                            "them is attributed to a raw value and a frame, so this is a result "
                            "about the policy extrapolating outside the 42,681 rows, not a "
                            "harness defect"
                            % (r["step"], r["episodes_off_band"], r["episodes"],
                               ",".join(str(c) for c in (r.get("base_columns_off_band") or []))
                               or "none",
                               ",".join(str(c) for c in (r.get("force_columns_off_band") or []))
                               or "none",
                               "-" if worst is None else "%.2f" % worst,
                               r.get("clamped_calls")))
    for r in timing:
        if not r.get("recorded") and not r.get("test_only"):
            warnings.append("step %d: the summary has no `timing` block, so this node is missing "
                            "from the measured per-episode cost the 160-episode decision is made "
                            "on" % r["step"])
    for s, d in sorted(docs.items()):
        cal = d.get("calibration") or {}
        if d.get("force_contract_channels") and list(d["force_contract_channels"]) != \
                list(cal.get("channels", d["force_contract_channels"])):
            warnings.append("step %d: the calibration file's channels and the frozen contract "
                            "channels differ" % s)
        if d.get("nan_aborts"):
            warnings.append("step %d: %d episode(s) aborted on a non-finite force reading"
                            % (s, d["nan_aborts"]))
        exp = d.get("expected_mae")
        if exp is not None:
            # the fixture flow run states the MAE its stub must produce, so the report can grade
            # its own force accounting instead of asking the reader to know the fixture's world
            mae = ((d.get("force") or {}).get("overall") or {}).get("mae")
            worst = max(mae) if mae else None
            verdict = "PASS" if worst is not None and abs(worst - abs(exp)) <= 1e-3 else "FAIL"
            warnings.append("step %d flow test: expected MAE %.4f N, worst channel %.4f N -- %s"
                            % (s, abs(exp), -1.0 if worst is None else worst, verdict))

    doc = {"generated": time.strftime("%FT%TZ"), "out": str(args.out),
           "seed_list": sorted(next(iter(seed_lists), ())),
           "nodes_on_disk": sorted(docs), "steps_requested": list(args.steps),
           "warnings": warnings, "rows": table_rows(args.steps, docs),
           "channel_table": channel_table(args.steps, docs),
           "band_table": band, "timing_table": timing,
           "run_metadata": {s: {k: d.get(k) for k in
                                ("backend", "policy", "config", "light_node", "calibration",
                                 "force_contract_channels", "wall_s")}
                            for s, d in sorted(docs.items())}}
    md = to_markdown(doc)
    if args.md:
        Path(args.md).write_text(md)
    if args.json:
        Path(args.json).write_text(json.dumps(doc, indent=2, sort_keys=True, default=str))
    print(md)
    # an unexplained off-band reading means the harness normalised something it should not have,
    # so every number in this report is suspect; the render still happens (the evidence is the
    # point) but the exit code says the batch must not be believed
    unexplained = sum(int(r.get("unexplained") or 0) for r in band if r.get("recorded"))
    print(json.dumps({"nodes_in_report": len([r for r in doc["rows"] if r.get("episodes")]),
                      "warnings": len(warnings), "unexplained_proprio": unexplained}))
    return 1 if unexplained else 0


if __name__ == "__main__":
    sys.exit(main())
