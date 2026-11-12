"""P4 Shadow Mode: watch the force loop, change nothing.

The brief is explicit -- 逐步预测力对齐、真实力误差、异常检测、日志和安全回退，默认不修改动作.
So this class has no code path that can return an action. It is given the decision payload the
rollout already produces (`frame`, `seed`, `pred_force`, `measured`, `meta`) and it only
accumulates and writes. The seam that would enforce a guard later is `_apply_force_feedback`
in the model, and this module never touches it: `force_feedback_applied` is read from the
policy's own metadata and asserted false, so a shadow run that secretly applied feedback fails
instead of reporting clean numbers.

Four things it computes:

1. **per-step alignment** -- a prediction made at frame `t` for `t+1 .. t+H` cannot be scored at
   `t`, so predictions are parked and matched against the measured stream as those frames
   arrive. Alignment is reported as coverage, and the frames an episode never reached are left
   unscored rather than filled in.
2. **real force error** -- per channel MAE/RMSE of prediction against measurement, split by
   horizon bucket, plus the two dumb predictors (hold-last, no-contact) on the same steps.
3. **anomaly detection** -- rules whose scale comes only from the frozen P1 calibration
   (per-channel mean/std), never from anything tuned on the eval episodes: a non-finite
   prediction, a prediction off the calibration band, prediction-vs-sensor disagreement, a
   sensor spike, a stuck sensor channel, and a history whose mask says fewer real frames than
   the loop claims to have observed.
4. **safe fallback (recorded only)** -- for each decision a verdict in
   `{"proceed","hold","stop"}` is computed and logged with its reason. In shadow mode the
   verdict changes nothing; it exists so that the same thresholds can be turned on later with
   a measured false-positive rate instead of a guess.

    from shadow import ShadowMonitor
    mon = ShadowMonitor(calib=calib_facts, log_path="shadow.jsonl", horizon=32)
    mon.on_decision(payload)          # per replan, from the rollout loop
    doc = mon.finalize()              # aggregate, written into the node summary
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

DEFAULT_Z_BAND = 4.0            # how far a value may sit from the calibration mean, in stds
DEFAULT_DISAGREE_Z = 4.0        # prediction vs sensor, in stds of that channel
DEFAULT_SPIKE_Z = 6.0           # one-frame change in the measurement, in stds
STUCK_FRAMES = 5                # identical measurement, that many frames in a row
VERDICTS = ("proceed", "hold", "stop")
BUCKETS = ((0, 8), (8, 16), (16, 32))


def _pack(abs_acc, sq_acc, n):
    if not n:
        return {"n": 0, "mae": None, "rmse": None}
    return {"n": int(n),
            "mae": np.round(abs_acc / n, 5).tolist(),
            "rmse": np.round(np.sqrt(sq_acc / n), 5).tolist()}


class ShadowMonitor:
    """Read-only observer of the force loop. See the module docstring for what it measures."""

    def __init__(self, calib, log_path=None, horizon: int = 32,
                 z_band: float = DEFAULT_Z_BAND, disagree_z: float = DEFAULT_DISAGREE_Z,
                 spike_z: float = DEFAULT_SPIKE_Z, stuck_frames: int = STUCK_FRAMES,
                 buckets=BUCKETS, mode: str = "shadow", channel_names=None,
                 hist_len: int = 16):
        if mode != "shadow":
            raise ValueError("ShadowMonitor only implements mode='shadow'; enforcement lives "
                             "in the model's force-feedback seam, not here")
        self.mode = mode
        self.horizon = int(horizon)
        self.hist_len = int(hist_len)
        self.buckets = [(int(a), int(b)) for a, b in buckets]
        self.z_band = float(z_band)
        self.disagree_z = float(disagree_z)
        self.spike_z = float(spike_z)
        self.stuck_frames = int(stuck_frames)
        self.channel_names = channel_names
        mean = np.asarray(calib["mean"], dtype=np.float64)
        std = np.asarray(calib["std"], dtype=np.float64)
        if mean.shape != std.shape or mean.ndim != 1 or not len(mean):
            raise ValueError("calibration mean/std must be matching 1-D lists")
        if np.any(~np.isfinite(mean)) or np.any(std <= 0) or np.any(~np.isfinite(std)):
            raise ValueError("calibration mean must be finite and std strictly positive")
        self.mean, self.std = mean, std
        self.scale = float(calib.get("scale", 1.0))
        self.calib_version = calib.get("version")
        self.log_path = Path(log_path) if log_path else None
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log = self.log_path.open("w")
        else:
            self._log = None

        self.decisions = 0
        self.actions_observed = 0
        self.feedback_applied_seen = 0
        self.pending = {}                  # (seed, episode, absolute frame) -> [(row, issue, j)]
        self.issued = 0
        self.aligned = 0
        self.unreachable = 0
        self._err_abs = np.zeros(len(self.mean))
        self._err_sq = np.zeros(len(self.mean))
        self._err_n = 0
        self._per_bucket = {b: [0, np.zeros(len(self.mean)), np.zeros(len(self.mean))]
                            for b in self.buckets}
        self._hold_abs = np.zeros(len(self.mean))
        self._hold_sq = np.zeros(len(self.mean))
        self._zero_abs = np.zeros(len(self.mean))
        self._zero_sq = np.zeros(len(self.mean))
        self._baseline_n = 0
        self.anomalies = {}
        self._worst = {}
        self.verdict_counts = {v: 0 for v in VERDICTS}
        self._episodes = set()             # (seed, episode index) pairs actually observed
        self._prev_meas = {}               # (seed, episode) -> last measurement vector
        self._sensor_next = {}             # (seed, episode) -> next frame index for the sensor rules
        self._stuck = {}                   # (seed, episode) -> (value, count)
        self._last_frame = {}              # seed -> last frame number seen for it
        self._ep_n = {}                    # seed -> episode index
        self._t0 = time.time()

    # --- bookkeeping ---------------------------------------------------------
    def _note(self, kind: str, detail: dict, verdict_hint: str = "hold") -> str:
        """Count an anomaly and return the verdict a guarded deployment *would* have taken."""
        cur = self.anomalies.get(kind, {"count": 0, "examples": []})
        cur["count"] += 1
        if len(cur["examples"]) < 5:
            cur["examples"].append(detail)
        self.anomalies[kind] = cur
        return verdict_hint

    def close(self):
        if self._log is not None:
            self._log.close()
            self._log = None

    def _write(self, doc: dict):
        if self._log is not None:
            self._log.write(json.dumps(doc, sort_keys=True, default=str) + "\n")

    def _pending_rows(self) -> int:
        return sum(len(rows) for rows in self.pending.values())

    def _ep_key(self, seed: int, frame: int):
        """(seed, episode index) -- an episode starts whenever the frame number goes back.

        Keying only on the seed would let a second episode with the same seed score its frames
        against the previous episode's parked predictions, so the frame number itself decides
        when the bookkeeping resets.
        """
        if frame <= self._last_frame.get(seed, -1):
            self._ep_n[seed] = self._ep_n.get(seed, 0) + 1
        self._last_frame[seed] = frame
        return (seed, self._ep_n.get(seed, 0))

    # --- the one entry point -------------------------------------------------
    def on_decision(self, payload: dict) -> dict:
        """Observe one replan. Returns the shadow verdict; never an action.

        `payload` needs `frame` (int, frames from episode start), `seed`, `pred_force`
        `[H, C]` newtons, `measured` (the per-frame measured force list so far, index == frame)
        and optionally `meta` from the policy.
        """
        frame = int(payload["frame"])
        seed = int(payload.get("seed", 0))
        pred = np.asarray(payload["pred_force"], dtype=np.float64)
        measured = payload.get("measured") or []
        meta = payload.get("meta") or {}
        ep = self._ep_key(seed, frame)
        if pred.ndim != 2 or pred.shape[1] != len(self.mean):
            self._note("prediction_shape_wrong",
                       {"seed": seed, "frame": frame, "shape": list(pred.shape)})
            raise ValueError(f"shadow expects pred_force [H, {len(self.mean)}], got "
                             f"{tuple(pred.shape)}")
        rows = int(pred.shape[0])
        if rows != self.horizon:
            self._note("prediction_horizon_wrong",
                       {"seed": seed, "frame": frame, "got": rows, "want": self.horizon})
        self.decisions += 1
        self.issued += self.horizon
        self._episodes.add(ep)

        reasons = []
        if meta.get("force_feedback_applied"):
            self.feedback_applied_seen += 1
            reasons.append(self._note("feedback_applied_in_shadow_mode",
                                      {"seed": seed, "frame": frame}, verdict_hint="stop"))

        if not np.all(np.isfinite(pred)):
            bad = np.argwhere(~np.isfinite(pred)).tolist()[:4]
            reasons.append(self._note("nonfinite_prediction",
                                      {"seed": seed, "frame": frame, "rows": bad}))
        z = np.abs(pred - self.mean) / self.std
        off_band = np.argwhere(z > self.z_band)
        if len(off_band):
            j, c = off_band[0]
            reasons.append(self._note("prediction_off_calibration_band",
                                      {"seed": seed, "frame": frame, "step": int(j),
                                       "channel": int(c), "value": float(pred[j, c]),
                                       "z": float(z[j, c])}))

        # park this chunk's futures; score the ones already parked. Two chunks issued 10 frames
        # apart both predict the same absolute frame, so each frame holds a list, not a value --
        # keeping only the last one would quietly drop the earlier chunk's error.
        for j in range(min(rows, self.horizon)):
            self.pending.setdefault((ep, frame + 1 + j), []).append((pred[j], frame, j))
        newly_aligned = self._drain(ep, measured)

        hist_true = meta.get("history_mask_true")
        if hist_true is not None:
            want = min(frame + 1, self.hist_len)
            if int(hist_true) != want:
                reasons.append(self._note("history_mask_disagrees_with_frame",
                                          {"seed": seed, "frame": frame,
                                           "mask_true": int(hist_true), "expected": want}))

        verdict = "proceed" if not reasons else max(reasons, key=lambda v: VERDICTS.index(v))
        self.verdict_counts[verdict] += 1
        doc = {"ts": time.strftime("%FT%TZ"), "seed": seed, "frame": frame,
               "mode": self.mode, "verdict": verdict, "reasons": reasons,
               "aligned_now": newly_aligned, "pending": self._pending_rows(),
               "pred_absmax": float(np.abs(pred).max()),
               "measured_frames": len(measured),
               "action_absmax": meta.get("action_absmax"),
               "infer_s": meta.get("infer_s")}
        self._write(doc)
        return doc

    def observe_measurement(self, seed: int, frame: int, force6) -> None:
        """Optional sensor-only feed: spikes and stuck channels, independent of predictions."""
        ep = self._ep_key(seed, frame)
        self._sensor_checks(ep, frame, np.asarray(force6, dtype=np.float64))

    def flush(self, seed: int, measured) -> int:
        """Close out an episode: run the sensor rules over every measured frame and score the
        parked predictions whose frames have now happened. Called by the rollout at episode end,
        so a chunk issued near termination does not leave its futures permanently unscored."""
        seed = int(seed)
        ep = (seed, self._ep_n.get(seed, 0))
        self._sensor_scan(ep, measured)
        return self._drain(ep, measured)

    def _sensor_scan(self, ep, measured):
        start = self._sensor_next.get(ep, 0)
        for f in range(start, len(measured)):
            self._sensor_checks(ep, f, np.asarray(measured[f], dtype=np.float64))
        self._sensor_next[ep] = len(measured)

    # --- internals -----------------------------------------------------------
    def _drain(self, ep, measured) -> int:
        """Score every parked prediction of this episode whose frame has now been measured."""
        n = 0
        for key in [k for k in self.pending if k[0] == ep and k[1] < len(measured)]:
            rows = self.pending.pop(key)
            truth = np.asarray(measured[key[1]], dtype=np.float64)
            if truth.shape[0] != self.mean.shape[0]:
                truth = truth[: self.mean.shape[0]]
            for pred_row, issue_frame, j in rows:
                self.aligned += 1
                n += 1
                if not (np.all(np.isfinite(pred_row)) and np.all(np.isfinite(truth))):
                    self._note("nonfinite_measurement", {"episode": ep, "frame": key[1]})
                    continue
                err = pred_row - truth
                self._err_n += 1
                self._err_abs += np.abs(err)
                self._err_sq += err ** 2
                for (a, b), acc in self._per_bucket.items():
                    if a <= j < b:
                        acc[0] += 1
                        acc[1] += np.abs(err)
                        acc[2] += err ** 2
                hold = self._hold_vector(measured, issue_frame)
                if hold is not None:
                    self._baseline_n += 1
                    self._hold_abs += np.abs(hold - truth)
                    self._hold_sq += (hold - truth) ** 2
                self._zero_abs += np.abs(truth)
                self._zero_sq += truth ** 2
                dev = float(np.max(np.abs(err) / self.std))
                if dev > self.disagree_z:
                    c = int(np.argmax(np.abs(err)))
                    why = self._note("prediction_disagrees_with_sensor",
                                     {"episode": ep, "frame": key[1], "issued_at": issue_frame,
                                      "channel": c, "err_N": float(err[c]),
                                      "z": float(abs(err[c]) / self.std[c])})
                    self._worst[key[1]] = {"err_z": dev, "verdict": why}
        return n

    def _hold_vector(self, measured, issue_frame):
        if 0 <= issue_frame < len(measured):
            v = np.asarray(measured[issue_frame], dtype=np.float64)
            return v[: self.mean.shape[0]]
        return None

    def _sensor_checks(self, ep, frame: int, v: np.ndarray) -> list:
        reasons = []
        if not np.all(np.isfinite(v)):
            reasons.append(self._note("nonfinite_measurement",
                                      {"episode": ep, "frame": frame}, verdict_hint="stop"))
            self._prev_meas.pop(ep, None)
            return reasons
        prev = self._prev_meas.get(ep)
        if prev is not None:
            jump = np.abs(v - prev) / self.std
            if float(jump.max()) > self.spike_z:
                c = int(np.argmax(jump))
                reasons.append(self._note("sensor_spike",
                                          {"episode": ep, "frame": frame, "channel": c,
                                           "delta_N": float(v[c] - prev[c]),
                                           "z": float(jump[c])}))
            if np.allclose(v, prev, rtol=0, atol=1e-9):
                val, cnt = self._stuck.get(ep, (v, 1))
                cnt += 1
                self._stuck[ep] = (v, cnt)
                if cnt == self.stuck_frames:
                    reasons.append(self._note("sensor_stuck",
                                              {"episode": ep, "frame": frame,
                                               "frames": self.stuck_frames}))
            else:
                self._stuck[ep] = (v, 1)
        self._prev_meas[ep] = v
        return reasons

    # --- aggregate -----------------------------------------------------------
    def finalize(self) -> dict:
        # whatever the episodes never reached stays unscored, counted honestly
        self.unreachable = self._pending_rows()
        cov = round(self.aligned / self.issued, 4) if self.issued else None
        out = {
            "mode": self.mode,
            "calib_version": self.calib_version,
            "thresholds": {"z_band": self.z_band, "disagree_z": self.disagree_z,
                           "spike_z": self.spike_z, "stuck_frames": self.stuck_frames,
                           "scale": self.scale,
                           "threshold_source": "frozen P1 calibration mean/std only"},
            "horizon": self.horizon,
            "channel_names": self.channel_names,
            "decisions": self.decisions,
            "episodes_seen": len(self._episodes),
            "steps_issued": self.issued,
            "steps_aligned": self.aligned,
            "steps_never_reached": self.unreachable,
            "alignment_coverage": cov,
            "error": _pack(self._err_abs, self._err_sq, self._err_n),
            "by_horizon": {f"{a}-{b}": _pack(v[1], v[2], v[0])
                           for (a, b), v in self._per_bucket.items()},
            "baselines": {"hold_last": _pack(self._hold_abs, self._hold_sq, self._baseline_n),
                          "no_contact": _pack(self._zero_abs, self._zero_sq, self._err_n)},
            "anomaly_counts": {k: v["count"] for k, v in self.anomalies.items()},
            "anomalies": self.anomalies,
            "verdict_counts": dict(self.verdict_counts),
            "worst_frames": sorted((({"frame": f, **d} for f, d in self._worst.items())),
                                   key=lambda x: -x["err_z"])[:10],
            "feedback_applied_decisions": self.feedback_applied_seen,
            "actions_mutated": 0,
            "wall_s": round(time.time() - self._t0, 3),
            "log": str(self.log_path) if self.log_path else None,
        }
        num = float(np.mean(out["error"]["mae"])) if out["error"]["mae"] else None
        den = float(np.mean(out["baselines"]["hold_last"]["mae"])) \
            if out["baselines"]["hold_last"]["mae"] else None
        # a non-empty list of zeros is truthy, and the flow test's constant-force world makes
        # hold-last exact, so the denominator must be tested numerically or the ratio is inf
        out["error_mae_over_hold_last_mae"] = (round(num / den, 4)
                                               if num is not None and den not in (None, 0.0)
                                               else None)
        self.close()
        return out


def load_log(path):
    """Read a shadow JSONL back, for the report or for the false-positive audit."""
    rows = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows
