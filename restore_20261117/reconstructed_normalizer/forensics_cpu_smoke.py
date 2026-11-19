"""CPU smoke for the Phase-4 panel write path -- no model, no GPU, no MuJoCo.

It exercises exactly the two things the GPU runs depend on and the pod environment can break:
  1. the runtime PanelRecorder backend override (cv2/mp4v) actually writes a decodable mp4, and
  2. record40k.PanelSink.add_frame accepts a real-schema obs (front/left_wrist 256x256 uint8,
     18-wide observation.force) and the ann it builds passes panelvideo40k's frame-sync checks.

Run on the pod:  /opt/venv/bin/python forensics_cpu_smoke.py
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

EVAL40K = "/workspace/eval_restore_20261117/deploy/stage_40k/dev/eval40k"
RECON = "/workspace/eval_restore_20261117/reconstructed_normalizer"
SIM = "/workspace/eval_restore_20261117/deploy/a3_dual_arm_sim/src"
for p in (RECON, EVAL40K, SIM):
    if p not in sys.path:
        sys.path.insert(0, p)

import run_forensics40k as F  # noqa: E402

PV = F.install_cv2_recorder()
import record40k as R  # noqa: E402
from a3_dual_arm_sim.contracts import (  # noqa: E402
    FRONT_IMAGE, LEFT_WRIST_IMAGE, RIGHT_WRIST_IMAGE, FORCE, STATE)


def main():
    tmp = Path(tempfile.mkdtemp(prefix="forensics_smoke_"))
    N = 15
    rng = np.random.default_rng(0)
    panel = R.PanelSink(tmp / "smoke_panel.mp4", fps=20, seed=11, step=40000,
                        total_frames=N + 1, replan=10)
    # chain_fn left as None -> the panel draws no action-chain rows; that path is covered by the
    # archived panelsink_selftest. Here we only prove the write + frame-sync + decodability.

    def obs_for(f):
        img = (rng.random((256, 256, 3)) * 255).astype("uint8")
        force = np.zeros(18, np.float32)
        force[0:3] = [5.0, 6.0, 7.0]          # left wrist force
        force[12:16] = [1.0, 2.0, 3.0, 4.0]   # four fingertip pads 12/13/14/15
        state = np.zeros(16, np.float32)
        state[7] = 0.9 - 0.1 * (f % 5)
        return {FRONT_IMAGE: img, LEFT_WRIST_IMAGE: img.copy(),
                RIGHT_WRIST_IMAGE: img.copy(), FORCE: force, STATE: state}

    def rec_for(f):
        h = [0.0] * 14
        h[6] = 1.0
        h[13] = -1.0
        return {"frame": f, "handed_14d": h, "requested_14d": h, "safety_reason": None,
                "terminated": False, "cookies_in_target": 0, "success": False}

    p14 = np.zeros(14)
    p14[11], p14[12] = 1.0, 2.0
    panel.add_frame(obs_for(0),
                    {"frame": 0, "handed_14d": None, "safety_reason": None,
                     "terminated": False, "cookies_in_target": None}, p14)
    for f in range(1, N + 1):
        panel.add_frame(obs_for(f), rec_for(f), p14)
    stats = panel.close()

    import cv2
    cap = cv2.VideoCapture(stats["path"])
    got = 0
    first = None
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if first is None:
            first = fr.copy()
        got += 1
    cap.release()

    # camera recovery: the composed frame's front camera is the source 256x256 enlarged 2x by
    # replication, so taking every other row/col from the top-left 512x512 must recover a 256x256
    # block whose checker pattern is intact. We only assert size here (exact pixel parity against a
    # known source is the archived panelvideo selftest's job).
    checks = {
        "backend_cv2_mp4v": stats["backend"] == "cv2/mp4v",
        "frames_written_eq_Nplus1": stats["frames_written"] == N + 1,
        "decoded_eq_written": got == stats["frames_written"],
        "canvas_wh_1664x576": stats["wh"] == [2 * PV.CAM_W + PV.PANEL_W, PV.PANEL_H],
        "mp4_nonempty": stats["bytes"] > 0,
        "decoded_frame_shape": None if first is None else list(first.shape),
        "pad_trace_samples": stats.get("pad_trace_samples"),
        "pad_channel_columns": stats.get("pad_channel_columns"),
        "notes": stats.get("notes"),
    }
    ok = all([checks["backend_cv2_mp4v"], checks["frames_written_eq_Nplus1"],
              checks["decoded_eq_written"], checks["canvas_wh_1664x576"], checks["mp4_nonempty"]])
    print(json.dumps({"verdict": "PASS" if ok else "FAIL", "checks": checks}, indent=2,
                     default=str))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())