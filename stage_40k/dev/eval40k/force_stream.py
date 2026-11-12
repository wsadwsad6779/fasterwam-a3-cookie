"""Rolling force history for inference, built to match the training windows exactly.

`A3VideoDataset.force_windows` gives the model the last `hist_len` raw-newton samples
*ending at* the current frame, left-padded with zeros, with the mask true only where real
samples exist; the future it supervises is `s+1 .. s+32`, right-padded. If the eval loop
right-pads instead, or lets a history straddle an episode boundary, the model sees an input
distribution it was never trained on and the measured success rate describes that mismatch
rather than the policy. So the padding rule is written down once, here, and `flow_test.py`
compares it against a hand-built expectation of the first four frames of an episode.

Nothing in this module touches the model or the simulator: it takes the 18-wide force
vector the A3 env puts in `obs["observation.force"]` -- the same array
`_r5_stage/build_force_arrays.py` copied out of the corpus parquet -- and selects the six
frozen channels.
"""
from __future__ import annotations

import numpy as np

CALIB_VERSION = "force-calib-v1"
CHANNELS = (0, 1, 2, 12, 13, 16)
CHANNEL_NAMES = ("L_wrist_fx", "L_wrist_fy", "L_wrist_fz",
                 "L_pad_inner", "L_pad_outer", "L_gripper_act")
HIST_LEN = 16
FORCE_DIM = 6
SCALE = 1.0
FORCE_OBS_KEY = "observation.force"


class ForceContractError(RuntimeError):
    """The calibration, the node, or the env disagrees with the frozen force contract."""


def check_calib(calib: dict, where: str = "calibration") -> dict:
    """Validate the calibration file the run is about to trust, and echo the facts."""
    if not isinstance(calib, dict):
        raise ForceContractError(f"{where}: expected a dict, got {type(calib).__name__}")
    problems = []
    if calib.get("version") != CALIB_VERSION:
        problems.append(f"version={calib.get('version')!r} != {CALIB_VERSION!r}")
    got_channels = [int(c) for c in (calib.get("channels") or [])]
    if got_channels != list(CHANNELS):
        problems.append(f"channels={got_channels} != {list(CHANNELS)}")
    for key in ("mean", "std"):
        v = calib.get(key)
        if not isinstance(v, list) or len(v) != FORCE_DIM:
            problems.append(f"{key} must list {FORCE_DIM} numbers, got {v!r}")
        elif not all(np.isfinite(v)) or key == "std" and min(v) <= 0.0:
            problems.append(f"{key} must be finite and std strictly positive, got {v!r}")
    scale = float(calib.get("scale", 1.0))
    if scale != SCALE:
        problems.append(f"scale={scale} != {SCALE}")
    if problems:
        raise ForceContractError(f"{where}: " + "; ".join(problems))
    return {"version": calib["version"], "channels": got_channels,
            "mean": [float(v) for v in calib["mean"]],
            "std": [float(v) for v in calib["std"]], "scale": scale}


def check_node_calibration(node_calib: dict, where: str = "checkpoint node") -> dict:
    """The fingerprint a node carries has to name the same geometry we are feeding it."""
    if not isinstance(node_calib, dict):
        raise ForceContractError(f"{where}: no force_calibration block to check")
    got = {k: node_calib.get(k) for k in ("hist_len", "force_dim", "scale")}
    want = {"hist_len": HIST_LEN, "force_dim": FORCE_DIM, "scale": SCALE}
    bad = {k: (got.get(k), v) for k, v in want.items() if got.get(k) != v}
    if bad:
        raise ForceContractError(
            f"{where}: force fingerprint disagrees with the frozen contract "
            + ", ".join(f"{k} node={a} expected={b}" for k, (a, b) in bad.items()))
    return got


def select_channels(force18, where: str = "obs") -> np.ndarray:
    """Take the six frozen channels out of the env's 18-wide force vector."""
    v = np.asarray(force18, dtype=np.float64).reshape(-1)
    if v.shape[0] < max(CHANNELS) + 1:
        raise ForceContractError(
            f"{where}: force vector has {v.shape[0]} channels, channel {max(CHANNELS)} "
            "is not there -- the env is not the A3 env this corpus was recorded from")
    if not np.all(np.isfinite(v)):
        raise ForceContractError(f"{where}: non-finite value in the force vector: {v.tolist()}")
    return v[list(CHANNELS)].astype(np.float32)


class ForceStream:
    """Last-`hist_len`-frames buffer, left-padded, exactly like the dataset's windows."""

    def __init__(self, hist_len: int = HIST_LEN, channels=CHANNELS):
        self.hist_len = int(hist_len)
        self.channels = tuple(int(c) for c in channels)
        self.n_frames = 0
        self.reset()

    def reset(self) -> None:
        self._buf = np.zeros((self.hist_len, len(self.channels)), dtype=np.float32)
        self.n_frames = 0

    def push(self, force18) -> np.ndarray:
        vec = select_channels(force18)[: len(self.channels)]
        self._buf = np.roll(self._buf, -1, axis=0)
        self._buf[-1] = vec
        self.n_frames += 1
        return vec

    def snapshot(self):
        """`(history[hist_len, 6] float32, mask[hist_len] bool)` for the next model call.

        The buffer is kept newest-last, so a full history needs no shuffling; the padded
        rows at the front are already zeros with mask false.
        """
        filled = min(self.n_frames, self.hist_len)
        if filled == 0:
            raise RuntimeError("ForceStream.snapshot() before any frame was pushed")
        mask = np.zeros(self.hist_len, dtype=bool)
        mask[self.hist_len - filled:] = True
        return self._buf.copy(), mask
