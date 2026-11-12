"""Score a predicted future-force chunk against what the simulator actually measured.

The model returns `force_future_newton`, the sampled future force trajectory for the frames
after the decision frame -- the same `s+1 .. s+32` the training loss supervised
(`A3VideoDataset.force_windows`). Scoring it needs one honest extra step: a chunk issued
near the end of an episode has futures that never happened, and those must be *left out*
rather than compared against a zero that the dataset would have padded in. Padding zeros in
the score would reward the model for predicting nothing wherever the episode ends early.

Two reference predictions ride along, because a bare MAE number means nothing on its own:
`persistence` repeats the force measured at the decision frame, and `zero` predicts no
contact at all. A force head that learned nothing about the future still beats persistence
only if it is flat and the world is flat, so the three numbers together say whether the
prediction has information in it.
"""
from __future__ import annotations

import numpy as np

from force_stream import CHANNEL_NAMES

DEFAULT_BUCKETS = ((0, 8), (8, 16), (16, 32))
CONTACT_THRESHOLD_N = 1.0


def _empty(shape):
    return {"n": 0, "abs": np.zeros(shape), "sq": np.zeros(shape)}


def _pack(acc, shape):
    out = {"n": int(acc["n"])}
    if acc["n"]:
        out["mae"] = np.round(acc["abs"] / acc["n"], 5).tolist()
        out["rmse"] = np.round(np.sqrt(acc["sq"] / acc["n"]), 5).tolist()
    else:
        out["mae"] = None
        out["rmse"] = None
    return out


class ForceErrorAccount:
    """Accumulates force-prediction error over every scored step of every episode."""

    def __init__(self, horizon: int = 32, buckets=DEFAULT_BUCKETS,
                 contact_threshold_n: float = CONTACT_THRESHOLD_N):
        self.horizon = int(horizon)
        self.buckets = [(int(a), int(b)) for a, b in buckets]
        self.contact_threshold_n = float(contact_threshold_n)
        ndim = len(CHANNEL_NAMES)
        self._all = _empty((ndim,))
        self._contact = _empty((ndim,))
        self._per_bucket = {b: _empty((ndim,)) for b in self.buckets}
        self._baseline = {"persistence": _empty((ndim,)), "zero": _empty((ndim,))}
        self.chunks = 0
        self.steps_issued = 0
        self.steps_scored = 0

    def _bucket_of(self, j: int):
        for a, b in self.buckets:
            if a <= j < b:
                return (a, b)
        return None

    def add_chunk(self, issue_frame: int, pred, measured) -> int:
        """Score one chunk. `measured[f]` is the 6-channel force at frame f, or None.

        Returns how many future steps were actually scored (0 when the episode ended
        before the first one, which is counted but never imputed).
        """
        p = np.asarray(pred, dtype=np.float64)
        if p.ndim != 2 or p.shape[1] != len(CHANNEL_NAMES):
            raise ValueError(f"predicted force chunk must be [H, {len(CHANNEL_NAMES)}], "
                             f"got {tuple(p.shape)}")
        if not np.all(np.isfinite(p)):
            raise ValueError("predicted force chunk carries NaN/Inf; refusing to score it")
        self.chunks += 1
        self.steps_issued += min(self.horizon, p.shape[0])
        if p.shape[0] < self.horizon:
            raise ValueError(f"model returned {p.shape[0]} future steps, the frozen horizon "
                             f"is {self.horizon}; a short chunk would silently rescale the "
                             "horizon buckets")
        base = None
        if 0 <= issue_frame < len(measured) and measured[issue_frame] is not None:
            base = np.asarray(measured[issue_frame], dtype=np.float64)
        scored = 0
        for j in range(self.horizon):
            f = int(issue_frame) + 1 + j
            if f >= len(measured) or measured[f] is None:
                continue
            truth = np.asarray(measured[f], dtype=np.float64)
            err = p[j] - truth
            if not np.all(np.isfinite(err)):
                continue
            scored += 1
            self._all["n"] += 1
            self._all["abs"] += np.abs(err)
            self._all["sq"] += err ** 2
            b = self._bucket_of(j)
            if b is not None:
                self._per_bucket[b]["n"] += 1
                self._per_bucket[b]["abs"] += np.abs(err)
                self._per_bucket[b]["sq"] += err ** 2
            if np.max(np.abs(truth)) >= self.contact_threshold_n:
                self._contact["n"] += 1
                self._contact["abs"] += np.abs(err)
                self._contact["sq"] += err ** 2
            if base is not None:
                e = base - truth
                self._baseline["persistence"]["n"] += 1
                self._baseline["persistence"]["abs"] += np.abs(e)
                self._baseline["persistence"]["sq"] += e ** 2
            e0 = -truth
            self._baseline["zero"]["n"] += 1
            self._baseline["zero"]["abs"] += np.abs(e0)
            self._baseline["zero"]["sq"] += e0 ** 2
        self.steps_scored += scored
        return scored

    def finalize(self) -> dict:
        ndim = len(CHANNEL_NAMES)
        out = {
            "channel_names": list(CHANNEL_NAMES),
            "horizon": self.horizon,
            "contact_threshold_n": self.contact_threshold_n,
            "chunks": self.chunks,
            "steps_issued": self.steps_issued,
            "steps_scored": self.steps_scored,
            "coverage": round(self.steps_scored / self.steps_issued, 4) if self.steps_issued else None,
            "overall": _pack(self._all, (ndim,)),
            "in_contact": _pack(self._contact, (ndim,)),
            "by_horizon": {f"{a}-{b}": _pack(v, (ndim,))
                           for (a, b), v in self._per_bucket.items()},
            "baselines": {k: _pack(v, (ndim,)) for k, v in self._baseline.items()},
        }
        # a non-empty list of zeros is truthy, so check the denominator itself: the flow test's
        # constant-force world makes persistence exact, and 0.00002 / 0.0 is inf, not a ratio
        means = {k: float(np.mean(v["mae"])) if v["mae"] else None
                 for k, v in [("model", out["overall"])] + list(out["baselines"].items())}
        num = means["model"]
        for ref, key in [("persistence", "mae_vs_persistence"), ("zero", "mae_vs_zero")]:
            out[key] = (round(num / means[ref], 4)
                        if num is not None and means[ref] not in (None, 0.0) else None)
        return out
