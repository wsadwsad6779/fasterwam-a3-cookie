"""Shadow Mode tested on synthetic decisions whose answers are known in advance.

P4 Shadow Mode has to be trustworthy before anyone is tempted to turn it on, and its
detections are thresholds, so the only honest test is to feed it values that are exactly on
each side of each rule and assert which ones fired. Every case below is built from the
calibration numbers themselves, not from anything the monitor computed, and the last case
checks the property the brief actually asks for: 默认不修改动作 -- the payload's arrays come
back byte-identical and the finalize document says `actions_mutated: 0`.

    cd /workspace/_v1_stage/dev/p4_shadow
    PYTHONDONTWRITEBYTECODE=1 nice -n 19 /opt/venv/bin/python -u shadow_selftest.py \
        --json ../out40k/shadow_selftest.json
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / "dev" / "eval40k"))

from force_stream import CHANNELS  # noqa: E402
from shadow import ShadowMonitor, load_log  # noqa: E402

NDIM = len(CHANNELS)
CALIB = {"version": "force-calib-v1", "channels": list(CHANNELS),
         "mean": [1.0] * NDIM, "std": [0.5] * NDIM, "scale": 1.0}
HORIZON = 32


def stream(fn, n_frames):
    """A measured-force list whose frame f is `fn(f)`, built independently of the monitor."""
    return [np.asarray(fn(f), dtype=np.float64) for f in range(n_frames)]


def flat(value):
    return np.full(NDIM, float(value), dtype=np.float64)


def decision(seed, frame, pred, measured, meta=None):
    return {"seed": seed, "frame": frame, "pred_force": pred, "measured": measured,
            "meta": meta or {}}


def new_monitor(log_path=None, **kw):
    kw.setdefault("horizon", HORIZON)
    return ShadowMonitor(calib=CALIB, log_path=log_path, **kw)


def results():
    checks = []
    doc = {"checks": checks, "cases": {}}

    # --- 1. a perfect predictor: aligned, zero error, no anomaly -------------------
    #
    # The "perfect" predictor is an oracle: each chunk predicts the frames the fixture really
    # went on to measure. It is fed the truncated observation list a rollout would have at that
    # moment, so alignment still has to happen the hard way, as futures arrive.
    mon = new_monitor()
    meas = stream(lambda f: flat(1.0) + 0.001 * f, 61)
    for frame in (0, 10, 20, 30, 40, 50):
        pred = np.stack([meas[min(frame + 1 + j, len(meas) - 1)] for j in range(HORIZON)])
        mon.on_decision(decision(7, frame, pred, meas[:frame + 1],
                                 {"history_mask_true": min(frame + 1, 16)}))
    mon.flush(7, meas)
    fin = mon.finalize()
    doc["cases"]["perfect"] = fin
    scored = [min(HORIZON, len(meas) - 1 - t) for t in (0, 10, 20, 30, 40, 50)]
    checks.append(("perfect predictions align every step the episode actually reached",
                   fin["steps_aligned"] == sum(scored) == 156))
    checks.append(("and count the unreachable futures as never scored, not as error-free",
                   fin["steps_never_reached"] == 6 * HORIZON - 156 == 36))
    checks.append(("perfect predictions score zero error on every channel",
                   fin["error"]["mae"] is not None and
                   bool(np.allclose(fin["error"]["mae"], 0.0, atol=1e-12))))
    checks.append(("perfect predictions raise no anomaly", fin["anomaly_counts"] == {}))
    checks.append(("alignment coverage counts only the frames the episode reached",
                   abs(fin["alignment_coverage"] - 156 / (6 * HORIZON)) < 1e-6))

    # --- 2. off-band prediction ---------------------------------------------------
    mon = new_monitor(z_band=4.0)
    meas = stream(lambda f: flat(1.0), 21)
    far = np.repeat((np.asarray(CALIB["mean"]) + 5.0 * np.asarray(CALIB["std"]))[None, :],
                    HORIZON, axis=0)
    d = mon.on_decision(decision(1, 0, far, meas[:1]))
    mon.flush(1, meas)
    fin = mon.finalize()
    checks.append(("a prediction 5 std off the calibration mean is flagged",
                   fin["anomaly_counts"].get("prediction_off_calibration_band") == 1))
    checks.append(("and the recorded fallback verdict is not 'proceed'", d["verdict"] != "proceed"))

    # --- 3. disagreement with the sensor -----------------------------------------
    #
    # mean + 5 std is 2.5 N above the measured 1.0 N: inside the (deliberately loosened)
    # calibration band, but 5 std of error -- past disagree_z -- so only the disagreement rule
    # may fire. Frames 1..11 happened, so exactly 11 scored steps.
    mon = new_monitor(disagree_z=4.0, z_band=100.0, stuck_frames=10 ** 6)
    meas = stream(lambda f: flat(1.0), 12)
    off = np.repeat((np.asarray(CALIB["mean"]) + 5.0 * np.asarray(CALIB["std"]))[None, :],
                    HORIZON, axis=0)
    mon.on_decision(decision(2, 0, off, meas[:1]))
    mon.flush(2, meas)
    fin = mon.finalize()
    checks.append(("prediction-vs-sensor disagreement beyond disagree_z is flagged",
                   fin["anomaly_counts"].get("prediction_disagrees_with_sensor") == 11))
    checks.append(("the disagreeing frames are kept as worst-frame evidence",
                   len(fin["worst_frames"]) > 0))
    checks.append(("error MAE equals the injected 2.5 N bias",
                   fin["error"]["mae"] is not None and
                   bool(np.allclose(fin["error"]["mae"], 2.5, atol=1e-9))))
    checks.append(("a constant sensor reading is not mistaken for a spike",
                   "sensor_spike" not in fin["anomaly_counts"]))

    # --- 4. non-finite prediction -------------------------------------------------
    mon = new_monitor()
    meas = stream(lambda f: flat(1.0), 12)
    bad = np.repeat(np.asarray(meas[0])[None, :], HORIZON, axis=0)
    bad[3, 2] = np.nan
    bad[9, 0] = np.inf
    mon.on_decision(decision(3, 0, bad, meas[:1]))
    mon.flush(3, meas)
    fin = mon.finalize()
    checks.append(("a NaN or Inf in the prediction is flagged",
                   fin["anomaly_counts"].get("nonfinite_prediction") == 1))
    checks.append(("and the affected steps are not silently scored",
                   fin["steps_aligned"] == 11 and fin["error"]["n"] == 9))

    # --- 5. sensor spike / stuck sensor ------------------------------------------
    #
    # One step up of 20 N = 40 std, then it stays there, so exactly one transition is a spike.
    mon = new_monitor(spike_z=6.0, stuck_frames=10 ** 6)
    base = flat(1.0)
    jump = flat(21.0)
    meas = stream(lambda f: base if f < 5 else jump, 20)
    mon.flush(4, meas)
    fin = mon.finalize()
    checks.append(("a 40-sigma sensor jump is flagged as a spike",
                   fin["anomaly_counts"].get("sensor_spike") == 1))

    mon = new_monitor(spike_z=6.0, stuck_frames=10 ** 6)
    meas = stream(lambda f: flat(1.0) + 1.0 * f, 20)      # 1 N/frame = 2 std, a real ramp
    mon.flush(4, meas)
    fin = mon.finalize()
    checks.append(("a 1 N per frame ramp is not a spike, so the rule is not 'any change'",
                   "sensor_spike" not in fin["anomaly_counts"]))

    mon = new_monitor(spike_z=6.0, stuck_frames=5)
    meas = stream(lambda f: flat(2.0), 12)
    mon.flush(5, meas)
    fin = mon.finalize()
    checks.append(("a sensor reporting the identical value for stuck_frames in a row is flagged",
                   fin["anomaly_counts"].get("sensor_stuck") == 1))

    # --- 6. feedback applied while in shadow mode --------------------------------
    mon = new_monitor()
    meas = stream(lambda f: flat(1.0), 12)
    pred = np.repeat(np.asarray(meas[0])[None, :], HORIZON, axis=0)
    d = mon.on_decision(decision(6, 0, pred, meas[:1], {"force_feedback_applied": True}))
    fin = mon.finalize()
    checks.append(("shadow mode notices if the model's feedback seam was used at all",
                   fin["anomaly_counts"].get("feedback_applied_in_shadow_mode") == 1))
    checks.append(("and that alone would have been a 'stop' verdict", d["verdict"] == "stop"))

    # --- 7. history mask disagreement --------------------------------------------
    mon = new_monitor()
    meas = stream(lambda f: flat(1.0), 12)
    pred = np.repeat(np.asarray(meas[0])[None, :], HORIZON, axis=0)
    mon.on_decision(decision(8, 10, pred, meas[:11], {"history_mask_true": 4}))
    fin = mon.finalize()
    checks.append(("a history mask that claims fewer real frames than the loop saw is flagged",
                   fin["anomaly_counts"].get("history_mask_disagrees_with_frame") == 1))

    # --- 8. wrong shapes refuse ---------------------------------------------------
    mon = new_monitor()
    meas = stream(lambda f: flat(1.0), 12)
    raised = None
    try:
        mon.on_decision(decision(9, 0, np.zeros((HORIZON, NDIM - 1)), meas[:1]))
    except ValueError as e:
        raised = str(e)
    checks.append(("a prediction with the wrong channel count is refused outright",
                   raised is not None))
    short = mon.on_decision(decision(9, 0, np.zeros((8, NDIM)), meas[:1]))
    fin = mon.finalize()
    checks.append(("a prediction shorter than the frozen horizon is flagged",
                   fin["anomaly_counts"].get("prediction_horizon_wrong") == 1))
    checks.append(("and the bad decision still gets logged", short["frame"] == 0))

    # --- 9. it never mutates what it is given ------------------------------------
    mon = new_monitor(log_path=HERE / "out" / "selftest_shadow.jsonl")
    meas = stream(lambda f: flat(1.0) + 0.01 * f, 41)
    pred = np.stack([flat(1.0 + 0.02 * f) for f in range(HORIZON)])
    payload = decision(10, 5, pred.copy(), meas, {"history_mask_true": 6,
                                                 "action_absmax": 0.7})
    before_pred = copy.deepcopy(payload["pred_force"])
    before_meas = copy.deepcopy(payload["measured"])
    returned = mon.on_decision(payload)
    mon.flush(10, meas)
    fin = mon.finalize()
    checks.append(("the predicted-force array is untouched by observation",
                   bool(np.array_equal(before_pred, payload["pred_force"]))))
    checks.append(("the measured-force stream is untouched by observation",
                   all(bool(np.array_equal(a, b)) for a, b in zip(before_meas, payload["measured"]))))
    checks.append(("the monitor returns a verdict document, never an action",
                   isinstance(returned, dict) and "action" not in returned))
    checks.append(("finalize states plainly that no action was mutated",
                   fin["actions_mutated"] == 0))

    # --- 10. the log is a real artifact ------------------------------------------
    rows = load_log(HERE / "out" / "selftest_shadow.jsonl")
    checks.append(("the JSONL log has exactly one line per decision", len(rows) == 1))
    checks.append(("verdict counts add up to the number of decisions",
                   sum(fin["verdict_counts"].values()) == fin["decisions"]))
    checks.append(("thresholds are reported as coming only from the frozen calibration",
                   fin["thresholds"]["threshold_source"] ==
                   "frozen P1 calibration mean/std only"))

    # --- 11. a mode that is not shadow is refused ---------------------------------
    refused = None
    try:
        ShadowMonitor(calib=CALIB, mode="enforce")
    except ValueError as e:
        refused = str(e)
    checks.append(("mode='enforce' is refused by this class, so 'default off' cannot drift",
                   refused is not None))
    bad_calib = None
    for calib in ({**CALIB, "std": [0.0] * NDIM}, {**CALIB, "mean": [float("nan")] * NDIM}):
        try:
            ShadowMonitor(calib=calib)
        except ValueError as e:
            bad_calib = str(e)
    checks.append(("a calibration with zero or NaN std is refused", bad_calib is not None))

    doc["n_checks"] = len(checks)
    doc["n_passed"] = sum(1 for _d, ok in checks if ok)
    doc["failed"] = [d for d, ok in checks if not ok]
    doc["all_passed"] = doc["n_passed"] == doc["n_checks"]
    for desc, ok in checks:
        print(("PASS  " if ok else "FAIL  ") + desc, flush=True)
    print(json.dumps({"n_checks": doc["n_checks"], "n_passed": doc["n_passed"],
                      "all_passed": doc["all_passed"], "failed": doc["failed"]}))
    return doc


def main():
    out = None
    for i, a in enumerate(sys.argv):
        if a == "--json" and i + 1 < len(sys.argv):
            out = Path(sys.argv[i + 1])
    doc = results()
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(doc, indent=2, sort_keys=True, default=str))
        print(json.dumps({"json": str(out)}))
    return 0 if doc["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
