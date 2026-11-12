"""How often do the P4 shadow rules fire on real force data when nothing is wrong?

`shadow_selftest.py` proves the rules fire on synthetic anomalies. It cannot answer the question
that decides whether enforcement is ever switched on: on the frozen expert corpus, with a
predictor that is *exactly right*, how many decisions would a guard have held or stopped? That
number is the detector's false-positive floor, and it is a floor because the only way to get an
anomaly-free prediction is to use the truth. Feeding the truth back as the prediction is not
cheating and not a model result -- it is an oracle used solely to measure the detector, and this
script never reports it as model accuracy.

Four things are audited, all with scale taken only from the frozen `force-calib-v1` mean/std:

* **off-band** (`z > z_band` on any of `horizon x channels` predicted values). The corpus *is* the
  distribution the calibration mean/std were measured on, so the per-value exceedance rate is a
  property of the force signal, not of a model -- and a decision tests up to 192 values at once,
  which is how a per-value rule becomes a per-decision rule.
* **disagrees-with-sensor** -- measured against a hold-last predictor, the honest no-skill
  reference: its errors are real, so its fire rate is a scale reference, not a false-positive rate.
* **spike / stuck** -- measurement-side rules that never see a prediction, run over every frame.
  A grasp onset is a genuine physical discontinuity, so a spike rule that fires on it cannot tell
  an anomaly from physics; the audit reports how many spike hits land on a closing gripper.
* **cross-check** -- the shipped `ShadowMonitor` is then run for real at its own defaults, and its
  counter must equal the analytic count, so the sweep cannot silently be a different rule.

    cd /workspace/_v1_stage/dev/p4_shadow
    PYTHONDONTWRITEBYTECODE=1 nice -n 19 python fp_audit.py --out out/fp_audit.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
STAGE = HERE.parent.parent
sys.path.insert(0, str(STAGE / "dev" / "eval40k"))
sys.path.insert(0, str(HERE))

from force_stream import CHANNEL_NAMES, check_calib  # noqa: E402
from shadow import (DEFAULT_DISAGREE_Z, DEFAULT_SPIKE_Z, DEFAULT_Z_BAND,  # noqa: E402
                    STUCK_FRAMES, ShadowMonitor)

ARR_DEFAULT = "/workspace/a3_data/a3_fw_force/arrays.npz"
CALIB_DEFAULT = "/workspace/a3_data/a3_fw_force/force_calib_v1.json"
GRIP_CLOSE_COL = 6          # the same gripper-closing proxy chan18_audit.py used


def _gaussian_tail(z):
    """P(|N(0,1)| > z) -- printed beside the empirical rate to show how non-Gaussian force is."""
    return float(2.0 * (1.0 - 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))))


def load_episodes(arr_path, channels):
    """Slice the corpus into per-episode (force6 newtons, action) rows -- read-only."""
    arr = np.load(arr_path, allow_pickle=True)
    if "force18" not in arr.files:
        raise SystemExit("fp_audit: %s has no force18 (has %s)" % (arr_path, sorted(arr.files)))
    f = np.asarray(arr["force18"], dtype=np.float64)
    starts = np.asarray(arr["ep_start"], dtype=np.int64)
    lens = np.asarray(arr["ep_len"], dtype=np.int64)
    act = np.asarray(arr["action"], dtype=np.float64) if "action" in arr.files else None
    eps = []
    for i, (s, L) in enumerate(zip(starts, lens)):
        s, L = int(s), int(L)
        if L < 2 or s + L > len(f):
            continue
        eps.append({"idx": i, "frames": L, "force": f[s:s + L][:, channels],
                    "action": None if act is None else act[s:s + L]})
    if not eps:
        raise SystemExit("fp_audit: no usable episodes in %s" % arr_path)
    return eps


def decision_frames(n_frames, horizon, replan):
    """Frames at which a decision's whole future stays inside the episode.

    A truncated window would need invented values to fill the prediction, and invented values are
    exactly what this audit must not feed to a detector.
    """
    last = n_frames - 1 - horizon
    return list(range(0, last + 1, replan)) if last >= 0 else []


def windows(eps, horizon, replan):
    """Yield (episode, decision frame) for every decision whose future is fully observed."""
    for ep in eps:
        for t in decision_frames(ep["frames"], horizon, replan):
            yield ep, t


def band_sweep(eps, mean, std, horizon, replan, bands):
    """Oracle predictions: the z of the true future at every decision, plus the sweep it implies."""
    per_value, per_decision = [], []
    for ep, t in windows(eps, horizon, replan):
        w = np.abs((ep["force"][t + 1:t + 1 + horizon] - mean) / std).reshape(-1)
        per_value.append(w)
        per_decision.append(float(w.max()))
    n_decisions = len(per_decision)
    if not n_decisions:
        return {"n_decisions": 0, "per_value_z": None, "sweep": {}}
    pool_v = np.concatenate(per_value)
    pool_d = np.asarray(per_decision)
    v_per_d = int(pool_v.size // n_decisions)
    return {
        "n_decisions": int(n_decisions),
        "n_predicted_values": int(pool_v.size),
        "values_per_decision": v_per_d,
        "per_value_abs_z": {"p50": float(np.percentile(pool_v, 50)),
                            "p90": float(np.percentile(pool_v, 90)),
                            "p99": float(np.percentile(pool_v, 99)),
                            "p99.9": float(np.percentile(pool_v, 99.9)),
                            "max": float(pool_v.max())},
        "per_decision_abs_z": {"p50": float(np.percentile(pool_d, 50)),
                               "p90": float(np.percentile(pool_d, 90)),
                               "max": float(pool_d.max())},
        "sweep": {f"{b:g}": {"per_value_rate": float((pool_v > b).mean()),
                             "decision_rate": float((pool_d > b).mean()),
                             "gaussian_per_value_rate": _gaussian_tail(b),
                             "iid_predicted_decision_rate":
                                 1.0 - (1.0 - float((pool_v > b).mean())) ** v_per_d}
                  for b in bands},
    }


def per_channel_z(eps, mean, std, channels):
    """Where each frozen channel's own distribution sits, in stds of that channel."""
    out = []
    for c in range(len(channels)):
        col = np.concatenate([(ep["force"][:, c] - mean[c]) / std[c] for ep in eps])
        a = np.abs(col)
        out.append({"channel": int(channels[c]), "name": CHANNEL_NAMES[c],
                    "p50": float(np.percentile(a, 50)), "p90": float(np.percentile(a, 90)),
                    "p99": float(np.percentile(a, 99)), "p99.9": float(np.percentile(a, 99.9)),
                    "max": float(a.max())})
    return out


def disagree_sweep(eps, mean, std, horizon, replan, thresholds):
    """Hold-last as the no-skill reference: how big is a real error, and what does that cost?"""
    per_decision = []
    for ep, t in windows(eps, horizon, replan):
        err = np.abs(ep["force"][t + 1:t + 1 + horizon] - ep["force"][t]) / std
        per_decision.append(float(err.max()))
    d = np.asarray(per_decision) if per_decision else np.zeros(0)
    return {"n_decisions": int(d.size),
            "predictor": "hold-last (no skill; its errors are real, so this is not a FP rate)",
            "per_decision_err_z": (None if not d.size else
                                   {"p50": float(np.percentile(d, 50)),
                                    "p90": float(np.percentile(d, 90)),
                                    "p99": float(np.percentile(d, 99)),
                                    "max": float(d.max())}),
            "sweep": {f"{t:g}": {"decision_rate": float((d > t).mean()) if d.size else None}
                      for t in thresholds}}


def jump_matrix(ep, std):
    x = ep["force"]
    return (np.abs(np.diff(x, axis=0)) / std).max(axis=1) if len(x) > 1 else np.zeros(0)


def sensor_sweep(eps, mean, std, spikes, stuck_frames):
    """Spike and stuck rules over every frame; neither of them looks at a prediction."""
    jumps, total_frames, close_hits = [], 0, {}
    runs, eps_with_stuck = [], 0
    for ep in eps:
        x = ep["force"]
        total_frames += len(x)
        jm = jump_matrix(ep, std)
        jumps.append(jm)
        if ep["action"] is not None and len(ep["action"]) == len(x):
            closing = ep["action"][1:, GRIP_CLOSE_COL] > 0
            m = min(len(closing), len(jm))
            for s in spikes:
                close_hits[f"{s:g}"] = close_hits.get(f"{s:g}", 0) + int(
                    np.count_nonzero(closing[:m] & (jm[:m] > s)))
        run = 1
        ep_long = 0
        for i in range(1, len(x)):
            if np.all(np.abs(x[i] - x[i - 1]) <= 1e-9):
                run += 1
            else:
                runs.append(run)
                ep_long = max(ep_long, run)
                run = 1
        runs.append(run)
        ep_long = max(ep_long, runs[-1])
        if ep_long >= stuck_frames:
            eps_with_stuck += 1
    jm_all = np.concatenate(jumps) if jumps else np.zeros(0)
    return {"n_frames": int(total_frames), "n_transitions": int(jm_all.size),
            "spike_abs_z": (None if not jm_all.size else
                            {"p50": float(np.percentile(jm_all, 50)),
                             "p99": float(np.percentile(jm_all, 99)),
                             "p99.9": float(np.percentile(jm_all, 99.9)),
                             "max": float(jm_all.max())}),
            "spike_sweep": {f"{s:g}": {"per_frame_rate":
                                       float((jm_all > s).mean()) if jm_all.size else None,
                                       "n_spikes": int((jm_all > s).sum()) if jm_all.size else 0,
                                       "on_closing_gripper": int(close_hits.get(f"{s:g}", 0))}
                            for s in spikes},
            "stuck": {"stuck_frames_threshold": int(stuck_frames),
                      "longest_run": int(max(runs)) if runs else 0,
                      "n_runs_ge_threshold": int(sum(1 for r in runs if r >= stuck_frames)),
                      "frames_in_runs_ge_threshold":
                          int(sum(r for r in runs if r >= stuck_frames)),
                      "episodes_with_a_stuck_run": int(eps_with_stuck),
                      "n_episodes": int(len(eps))}}


def live_cross_check(eps, calib_facts, horizon, replan, max_eps):
    """Run the shipped monitor itself with oracle predictions, at its own default thresholds."""
    use = eps[:max_eps] if max_eps else eps
    mon = ShadowMonitor(calib=calib_facts, log_path=None, horizon=horizon,
                        channel_names=CHANNEL_NAMES)
    analytic_off_band = 0
    for ep in use:
        x = ep["force"]
        seed = int(ep["idx"])
        for t in decision_frames(ep["frames"], horizon, replan):
            pred = x[t + 1:t + 1 + horizon]
            if float(np.abs((pred - np.asarray(calib_facts["mean"]))
                            / np.asarray(calib_facts["std"])).max()) > mon.z_band:
                analytic_off_band += 1
            mon.on_decision({"frame": int(t), "seed": seed, "pred_force": pred.copy(),
                             "measured": x[:t + 1].tolist(), "meta": {}})
        mon.flush(seed, x.tolist())
    fin = mon.finalize()
    return {"n_episodes": len(use), "decisions": fin["decisions"],
            "steps_issued": fin["steps_issued"], "steps_aligned": fin["steps_aligned"],
            "steps_never_reached": fin["steps_never_reached"],
            "alignment_coverage": fin["alignment_coverage"],
            "anomaly_counts": fin["anomaly_counts"], "verdict_counts": fin["verdict_counts"],
            "error_mae_N": fin["error"]["mae"], "error_n": fin["error"]["n"],
            "analytic_off_band_decisions": int(analytic_off_band)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arr", default=ARR_DEFAULT)
    ap.add_argument("--calib", default=CALIB_DEFAULT)
    ap.add_argument("--horizon", type=int, default=32)
    ap.add_argument("--replan", type=int, default=10)
    ap.add_argument("--bands", default="3,4,5,6,8,10,12")
    ap.add_argument("--disagree", default="4,6,8,12")
    ap.add_argument("--spikes", default="4,6,8,12")
    ap.add_argument("--max-eps", type=int, default=20,
                    help="episodes fed to the live monitor cross-check (0 = all)")
    ap.add_argument("--out", default=str(HERE / "out" / "fp_audit.json"))
    args = ap.parse_args()

    t0 = time.time()
    calib_raw = json.loads(Path(args.calib).read_text())
    calib_facts = check_calib(calib_raw, where="fp_audit calibration")
    channels = calib_facts["channels"]
    mean = np.asarray(calib_facts["mean"], dtype=np.float64)
    std = np.asarray(calib_facts["std"], dtype=np.float64)
    if calib_raw.get("horizon") is not None and int(args.horizon) != int(calib_raw["horizon"]):
        raise SystemExit("fp_audit: --horizon %d disagrees with the calibration file's %s"
                         % (args.horizon, calib_raw["horizon"]))

    eps = load_episodes(args.arr, channels)
    bands = [float(b) for b in args.bands.split(",")]
    dis_thr = [float(b) for b in args.disagree.split(",")]
    spikes = [float(b) for b in args.spikes.split(",")]
    shipped = {"z_band": DEFAULT_Z_BAND, "disagree_z": DEFAULT_DISAGREE_Z,
               "spike_z": DEFAULT_SPIKE_Z, "stuck_frames": STUCK_FRAMES}

    band = band_sweep(eps, mean, std, args.horizon, args.replan, bands)
    dis = disagree_sweep(eps, mean, std, args.horizon, args.replan, dis_thr)
    sens = sensor_sweep(eps, mean, std, spikes, STUCK_FRAMES)
    live = live_cross_check(eps, calib_facts, args.horizon, args.replan, args.max_eps)
    agrees = bool(live["analytic_off_band_decisions"]
                  == live["anomaly_counts"].get("prediction_off_calibration_band", -1))

    doc = {"generated": time.strftime("%FT%TZ"), "arr": str(args.arr),
           "calib": str(args.calib), "calib_version": calib_facts["version"],
           "channels": channels, "channel_names": list(CHANNEL_NAMES),
           "mean": [float(v) for v in mean], "std": [float(v) for v in std],
           "n_episodes": len(eps), "n_frames": int(sum(ep["frames"] for ep in eps)),
           "n_episodes_shorter_than_horizon":
               int(sum(1 for ep in eps if decision_frames(ep["frames"], args.horizon,
                                                          args.replan) == [])),
           "horizon": args.horizon, "replan": args.replan,
           "shipped_thresholds": shipped,
           "oracle_band": band, "per_channel_abs_z": per_channel_z(eps, mean, std, channels),
           "hold_last_disagreement": dis, "sensor_rules": sens, "live_monitor": live,
           "cross_check": {"analytic_off_band_decisions": live["analytic_off_band_decisions"],
                           "monitor_off_band_count": live["anomaly_counts"].get(
                               "prediction_off_calibration_band", 0),
                           "agrees": agrees},
           "read_this_as": [
               "oracle_band.sweep[X].decision_rate is the share of replans a guard at z_band=X "
               "would have held even with a perfectly correct prediction",
               "hold_last_disagreement is a scale reference, not a false-positive rate",
               "spike/stuck counts are on expert data, so any hit is physics or quantisation, "
               "not a sensor fault"],
           "what_this_is_not": ["not a model result: the predictions are the measured truth",
                                "not training data or labels: nothing here is written back"],
           "wall_s": round(time.time() - t0, 2)}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2, sort_keys=True, default=str))

    print("episodes=%d frames=%d decisions=%d values/decision=%s"
          % (doc["n_episodes"], doc["n_frames"], band["n_decisions"],
             band.get("values_per_decision")))
    print("off-band sweep   [z  per-value  per-decision  iid-predicted  gaussian-per-value]")
    for b in bands:
        s = band["sweep"][f"{b:g}"]
        print("  z>%-4g  %10.5f  %13.4f  %13.4f  %16.3e"
              % (b, s["per_value_rate"], s["decision_rate"], s["iid_predicted_decision_rate"],
                 s["gaussian_per_value_rate"]))
    print("per-channel |z| p50/p99/p99.9/max:",
          [(r["name"], round(r["p50"], 2), round(r["p99"], 2), round(r["p99.9"], 2),
            round(r["max"], 2)) for r in doc["per_channel_abs_z"]])
    print("hold-last disagreement per decision:",
          {k: round(v["decision_rate"], 4) for k, v in dis["sweep"].items()},
          "| err_z p50/p90/p99/max:",
          None if dis["per_decision_err_z"] is None else
          [round(dis["per_decision_err_z"][k], 2) for k in ("p50", "p90", "p99", "max")])
    print("spike per frame [rate / n / on-closing-gripper]:",
          {k: (round(v["per_frame_rate"], 6), v["n_spikes"], v["on_closing_gripper"])
           for k, v in sens["spike_sweep"].items()},
          "| abs jump z max:", None if sens["spike_abs_z"] is None
          else round(sens["spike_abs_z"]["max"], 2))
    print("stuck:", sens["stuck"])
    print("live monitor: decisions=%d coverage=%s off_band=%s verdicts=%s mae_N=%s"
          % (live["decisions"], live["alignment_coverage"],
             live["anomaly_counts"].get("prediction_off_calibration_band", 0),
             live["verdict_counts"], live["error_mae_N"]))
    print("cross_check agrees=%s (analytic %d vs monitor %d)"
          % (agrees, live["analytic_off_band_decisions"],
             live["anomaly_counts"].get("prediction_off_calibration_band", -1)))
    print("out: %s" % out)
    return 0 if agrees else 1


if __name__ == "__main__":
    sys.exit(main())
