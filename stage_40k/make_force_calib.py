"""Measure the force calibration the P2/P3 stream normalizes with.

Writes /workspace/a3_data/a3_fw_force/force_calib_v1.json:

    {mean, std, channels, scale, version, ...provenance}

`runtime._load_force_norm_stats` refuses a calibration without `version`, without
`channels`, or whose channels differ from what the model gathers, so this script
has to record all of it.

Why one global mean/std and not a per-episode zero point: a roll-out has no
episode boundary to average over, and the earlier "first 20 frames of each
episode" estimator measured up to a 5.017-sigma progressive bias on `L_wrist_fy`
by the sixth frame -- it is not available at inference step 0. A constant taken
from the corpus is causal in the only sense that matters here: it is the same
number before the first frame and after the last.

Usage (on the instance):
    python make_force_calib.py [out.json]
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

FORCE_DIR = Path("/workspace/a3_data/a3_fw_force")
V1G_ARRAYS = Path("/workspace/a3_data/a3_fw_v1g/arrays.npz")
CHANNELS = [0, 1, 2, 12, 13, 16]
CHANNEL_NAMES = ["L_wrist_fx", "L_wrist_fy", "L_wrist_fz",
                 "L_pad_inner", "L_pad_outer", "L_gripper"]
VERSION = "force-calib-v1"


def die(msg):
    raise SystemExit("make_force_calib: " + msg)


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else FORCE_DIR / "force_calib_v1.json")
    src = FORCE_DIR / "arrays.npz"
    if not src.is_file():
        die("missing %s" % src)

    arr = np.load(src, allow_pickle=True)
    if "force18" not in arr.files:
        die("arrays.npz has no force18 (has %s)" % sorted(arr.files))
    f18 = np.asarray(arr["force18"], dtype=np.float64)
    if f18.ndim != 2 or f18.shape[1] < max(CHANNELS) + 1:
        die("force18 shape %s cannot supply channel %d" % (f18.shape, max(CHANNELS)))
    raw6 = f18[:, CHANNELS]
    if not np.isfinite(raw6).all():
        die("gathered force columns contain non-finite values")

    # The 14-dim corpus is what training actually reads, so the calibration must
    # describe the same rows. `ep_start/ep_len` are compared, not assumed.
    if V1G_ARRAYS.is_file():
        v1 = np.load(V1G_ARRAYS, allow_pickle=True)
        for key in ("ep_start", "ep_len"):
            if key not in v1.files or key not in arr.files:
                continue
            if not np.array_equal(np.asarray(v1[key]), np.asarray(arr[key])):
                die("%s disagrees between %s and %s" % (key, src, V1G_ARRAYS))
        if int(v1["proprio"].shape[1]) != 14:
            die("proprio in %s is %d wide, expected 14" % (V1G_ARRAYS, v1["proprio"].shape[1]))
        corpus = str(V1G_ARRAYS)
        n_rows_train = int(v1["proprio"].shape[0])
    else:
        die("missing %s: calibrating against a corpus the trainer does not read"
            % V1G_ARRAYS)
    if n_rows_train != f18.shape[0]:
        die("corpus has %d rows but force18 has %d" % (n_rows_train, f18.shape[0]))

    zero_ref = None
    stats_path = FORCE_DIR / "force_stats.json"
    if stats_path.is_file():
        stats = json.loads(stats_path.read_text())
        if "zero_load_ref_median" in stats:
            zero_ref = [float(v) for v in stats["zero_load_ref_median"]]
            if len(zero_ref) != len(CHANNELS):
                die("zero_load_ref_median has %d entries for %d channels"
                    % (len(zero_ref), len(CHANNELS)))

    mean = raw6.mean(axis=0)
    std = raw6.std(axis=0, ddof=0)
    # A zero std would turn the stream into an inf/NaN generator one contact spike
    # away, and a near-zero one would amplify sensor noise into huge "normalized"
    # values. Both are corpus bugs, so stop rather than clamp and hide them.
    if (std < 1e-3).any():
        bad = {CHANNEL_NAMES[i]: float(std[i]) for i in range(len(CHANNELS)) if std[i] < 1e-3}
        die("suspiciously flat channel(s): %s" % bad)

    payload = {
        "version": VERSION,
        "units": "newton (raw sensor values, no gravity compensation, no torque)",
        "channels": CHANNELS,
        "channel_names": CHANNEL_NAMES,
        "channel_note": (
            "Left wrist 3-axis force + left finger-pad normal forces + left gripper "
            "actuation force. Not a six-axis wrench: no torque channel is present, and "
            "the wrist components are in the flange frame."
        ),
        "mean": [float(v) for v in mean],
        "std": [float(v) for v in std],
        "scale": 1.0,
        "zero_load_ref_median": zero_ref,
        "n_rows": int(f18.shape[0]),
        "n_episodes": int(np.asarray(arr["ep_len"]).shape[0]) if "ep_len" in arr.files else None,
        "corpus": corpus,
        "source_arrays": str(src),
        "source_sha256": {
            "force_arrays": sha256(src),
            "corpus_arrays": sha256(V1G_ARRAYS),
        },
        "generator": "make_force_calib.py",
        "per_channel_absmax": [float(v) for v in np.abs(raw6).max(axis=0)],
        "per_channel_min": [float(v) for v in raw6.min(axis=0)],
        "per_channel_max": [float(v) for v in raw6.max(axis=0)],
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, out)

    back = json.loads(out.read_text())
    if back["mean"] != payload["mean"] or back["std"] != payload["std"]:
        die("round-trip mismatch after writing %s" % out)
    print(json.dumps({
        "wrote": str(out),
        "bytes": out.stat().st_size,
        "mean": [round(v, 6) for v in payload["mean"]],
        "std": [round(v, 6) for v in payload["std"]],
        "n_rows": payload["n_rows"],
        "sha256": payload["source_sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
