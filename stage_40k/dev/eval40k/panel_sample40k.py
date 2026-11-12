"""Render a watchable layout sample of the composed panel video, on this machine, without the pod.

The pod's JupyterLab channel answers 403, so no real rollout can be recorded right now. What can be
settled offline is the question the format request actually asks -- is the picture laid out the way
it was specified: front on the left, left_wrist in the middle, every number in a separate right-hand
panel that never covers a camera, the two fingertip forces shown as two independent large readings
with their own curves and a playhead, the historical 2x camera scale and 20 fps.

So this drives the *production* code path (`record40k.PanelSink`, the same `add_frame` a rollout
calls) with hand-made observations, and writes the mp4 the report will deliver. Two things it is not:
it is not evidence about the policy -- no weights are loaded and no simulator runs, so the force and
camera contents are invented (the action chain is the exception, see below) -- and it is not a
substitute for the seed 11/17 recordings, which are the ones that have to explain the arm flip and the
sustained saturation. Both facts are printed on the panel itself, as its first note, so a screenshot
cannot lose them.

What is invented was chosen to be legible rather than plausible: a quiet stretch, a left fingertip
pressing past the 20 N contact threshold, a right fingertip burst with an emergency stop, and a
release. The action chain is not invented: both regimes replay numbers logged by the real link and
transcribed into ROOT_CAUSE_40K_20261112.md section 1.1, in the units each stage really uses
(normalised units / LIBERO action units / the +-1 rate command `env.step` receives). An earlier
version of this file made up its own chain and put 44.37 in the normalised row, which is exactly the
error section 1.1 was written to kill -- that magnitude cannot appear before denormalisation.

    py -3.13 panel_sample40k.py            # writes dev/out40k/panel_layout_sample/
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import panelvideo40k as PV     # noqa: E402
import record40k as R          # noqa: E402

OUT = HERE.parent / "out40k" / "panel_layout_sample"
FRAMES = 300
REPLAN = 10
TOTAL = 600                      # the batch's episode length, so the axis looks like the real one
CONTACT_ON = R.CONTACT_ON_N
BURST_START, BURST_END = 170, 205

# each note has to fit the panel's text box (about 78 characters at font 11), because the composer
# now cuts a longer line with a visible ellipsis instead of letting it run off the edge
NOTES = (
    "LAYOUT SAMPLE: no weights, no sim; synthetic force/cameras; real seed 11/17 replace this",
)


def force(i):
    """An 18-wide `observation.force` in raw newtons, with the episode's four regimes."""
    f = np.zeros(18, dtype=np.float64)
    # wrist: the empty-load reading is fy ~= -10.7 N (the flange hang weight projected on the up axis)
    lean = 0.0 if i < 60 else min(1.0, (i - 60) / 120.0)
    f[0] = 0.6 * np.sin(i / 9.0) + 3.0 * lean
    f[1] = -10.72 + 1.2 * np.sin(i / 13.0) - 6.0 * lean
    f[2] = 0.15 * np.cos(i / 7.0)
    f[3], f[4], f[5] = 0.4 * lean, 0.2 * np.sin(i / 20.0), 0.1 * lean
    if 90 <= i < 160:                                   # left fingertip presses the cookie
        k = (i - 90) / 70.0
        f[12] = 2.0 + 58.0 * k * (2.0 - k)              # rises to ~58 N, concave
        f[13] = 1.0 + 24.0 * k
    elif i >= 160:
        f[12], f[13] = 6.0, 3.0
    if BURST_START <= i < BURST_END:                    # right fingertip: penetration burst
        k = (i - BURST_START) / (BURST_END - BURST_START)
        f[14] = 120.0 * np.sin(np.pi * k) ** 2
        f[15] = 70.0 * np.sin(np.pi * k) ** 1.5
    f[16] = 12.0 if i > 100 else 0.0                    # L gripper actuator
    f[17] = 4.0 * lean
    return f


def cameras(i):
    """Two 256x256 placeholders that move, so the 2x block and the no-overlay rule are visible."""
    from PIL import Image, ImageDraw

    x = int(28 + (196.0 * i / (FRAMES - 1)))
    front = Image.new("RGB", (PV.CAM_SRC, PV.CAM_SRC), (26, 30, 34))
    d = ImageDraw.Draw(front)
    d.rectangle([16, 150, 240, 240], outline=(70, 78, 86), width=2)          # the box
    d.ellipse([x - 9, 60, x + 9, 78], fill=(150, 120, 90))                   # held cookie
    d.line([(128, 0), (x, 66)], fill=(120, 128, 136), width=5)               # arm
    wrist = Image.new("RGB", (PV.CAM_SRC, PV.CAM_SRC), (18, 20, 24))
    d = ImageDraw.Draw(wrist)
    gap = max(4, 40 - int(26.0 * min(1.0, max(0.0, (i - 90) / 70.0))))
    d.rectangle([128 - gap - 26, 90, 128 - gap, 190], fill=(96, 102, 110))   # pad inner
    d.rectangle([128 + gap, 90, 128 + gap + 26, 190], fill=(96, 102, 110))   # pad outer
    if BURST_START <= i < BURST_END:
        d.rectangle([102, 84, 154, 196], outline=(210, 90, 70), width=3)     # contact flash
    return {"front": np.asarray(front), "left_wrist": np.asarray(wrist),
            "right_wrist": np.asarray(wrist)[:, ::-1]}


def chain_for_frame(i):
    """The action chain for this frame, replaying a logged witness rather than an invented number.

    Both regimes come from ROOT_CAUSE_40K_20261112.md section 1.1. `executed_row` is deliberately
    absent: the real recorder fills it (and `ik_cmd_row`) from `rec["handed_14d"]` /
    `rec["requested_14d"]` inside `PanelSink.add_frame`, so the sample feeds those keys the same way a
    rollout does instead of pushing them through the chain side door.
    """
    return dict(BURST if BURST_START <= i < BURST_END else QUIET)


# seed 17, the worst row the recorder logged end to end (chunk 26, row 31): model output in normalised
# units is the per-dimension |max| of that replan, the three later stages are that one row.
BURST = {
    "model_absmax_dims": [33.5, 27.5, 21.5, 24.125, 18.5, 15.875, 40.0],
    "model_over5_frac": [0.523, 0.441, 0.491, 0.451, 0.440, 0.072, 0.617],
    "denorm_row": [1.3268, 0.0580, -4.3467, 0.6686, -0.3432, 0.0595, 40.0],
    "pm1_row": [1.3268, 0.0580, -4.3467, 0.6686, -0.3432, 0.0595, 1.0],
    "executed_row": [1.0, 0.1159, -1.0, 1.0, 1.0, 1.0, -1.0, 0, 0, 0, 0, 0, 0, 1.0],
}
# seed 11: this seed never left the +-5 band and never saturated a column, so the four logged
# per-dimension maxima are episode-wide |max| values rather than one row. The executed gripper column
# follows the adapter's own rule, output[6] = -sign(action[6]) (action_adapter_v3.py:48).
QUIET = {
    "model_absmax_dims": [3.5781, 1.9453, 2.6094, 2.9844, 2.6250, 3.1562, 2.1875],
    "model_over5_frac": [0.0] * 7,
    "denorm_row": [0.2824, 0.3827, 0.4184, 0.0974, 0.0492, 0.0834, 2.1875],
    "pm1_row": [0.2824, 0.3827, 0.4184, 0.0974, 0.0492, 0.0834, 1.0],
    "executed_row": [0.5647, 0.7655, 0.8367, 0.4343, 0.4849, 0.4278, -1.0, 0, 0, 0, 0, 0, 0, 1.0],
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "LAYOUT_SAMPLE_synth_300f_1664x576_20fps.mp4"
    sink = R.PanelSink(path, fps=20, seed=0, step="LAYOUT SAMPLE", total_frames=TOTAL, replan=REPLAN,
                       extra_notes=list(NOTES))
    sink.chain_fn = lambda f: chain_for_frame(int(f))
    for i in range(FRAMES):
        f = force(i)
        if i == BURST_END:
            sink.emergency.append({"frame": i, "args": ("sample: injected stop",)})
        obs = {"observation.images.%s" % n: img for n, img in cameras(i).items()}
        obs["observation.force"] = f
        p14 = np.zeros(14, dtype=np.float64)
        p14[8:14] = [f[0], f[1], f[2], f[12] + 0.00012, f[13], f[16] + 0.01863]
        # frame 0 is the reset observation: the sink must see no action at all, because nothing has
        # been executed yet. Every later frame replays one control step, and `requested_14d` equals
        # `handed_14d` because validate_action raises rather than clips above +-1 (contracts.py:60-61).
        chain = chain_for_frame(i)
        cmd = None if i == 0 else chain["executed_row"]
        sink.add_frame(obs, {"frame": i, "handed_14d": cmd, "requested_14d": cmd,
                             "cookies_in_target": 0, "terminated": i >= BURST_END,
                             "safety_reason": ("sample: injected stop" if i >= BURST_END else None)},
                       p14)
    stats = sink.close()

    back = PV.read_frames(path)
    problems = []
    if len(back) != FRAMES or stats["frames_written"] != FRAMES:
        problems.append("decoded %d of %d written" % (len(back), stats["frames_written"]))
    if stats["wh"] != [2 * PV.CAM_W + PV.PANEL_W, PV.PANEL_H]:
        problems.append("geometry %s" % stats["wh"])
    if list(back[0].shape) != stats["wh"][::-1] + [3]:
        problems.append("decodes to %s" % (back[0].shape,))
    for tag, a, b in (("front", back[60][:PV.CAM_H, :PV.CAM_W], back[240][:PV.CAM_H, :PV.CAM_W]),
                      ("left_wrist", back[60][:PV.CAM_H, PV.CAM_W:2 * PV.CAM_W],
                       back[240][:PV.CAM_H, PV.CAM_W:2 * PV.CAM_W])):
        if float(np.abs(a.astype(int) - b.astype(int)).mean()) < 0.5:
            problems.append("the %s block never changes -- it is a still, not a camera" % tag)
    # The text block is where the two large fingertip readings live, so if the numbers really update
    # this region has to differ between a quiet frame, a pressing frame and the burst frame.
    px = 2 * PV.CAM_W
    digits = [int((back[i][6:306, px + 6:px + 330].max(axis=2) > 120).sum())
              for i in (10, 120, BURST_START + 12)]
    if len(set(digits)) < 3:
        problems.append("the reading region holds the same ink in 2 of 3 sampled frames: %s" % digits)
    # Each stage must stay inside its own units, or the panel's four lines stop meaning four things.
    # The executed row is the rate command the actuators take, so it cannot leave +-1; the stage before
    # it is the one that does, and that gap is the whole content of section 1.1.
    for name, ch in (("burst", BURST), ("quiet", QUIET)):
        if float(np.abs(ch["executed_row"]).max()) > 1.0 + 1e-12:
            problems.append("%s executed row left the +-1 rate band" % name)
        if ch["pm1_row"][6] not in (-1.0, 1.0):
            problems.append("%s pm1 gripper col is %s, pm1 pins it to +-1" % (name, ch["pm1_row"][6]))
        if ch["denorm_row"][:6] != ch["pm1_row"][:6]:
            problems.append("%s pm1 changed a column other than the gripper" % name)
    if float(np.abs(BURST["pm1_row"]).max()) <= 1.0:
        problems.append("the burst chain shows no out-of-band stage, so it proves nothing about units")
    still = []
    for i in (0, 110, BURST_START + 12, BURST_END + 40, FRAMES - 1):
        p = OUT / ("sample_frame_%03d.png" % i)
        import cv2

        cv2.imwrite(str(p), back[i][:, :, ::-1])
        still.append({"frame": i, "png": str(p), "bytes": p.stat().st_size})

    meta = {"purpose": ("layout/format review only -- synthetic force and camera content, no model "
                        "weights loaded and no MuJoCo state, so nothing here describes the policy or a "
                        "real contact event. The action chain is not synthetic: it replays the logged "
                        "stage numbers transcribed into ROOT_CAUSE_40K_20261112.md section 1.1."),
            "produced_by": "panel_sample40k.py through record40k.PanelSink.add_frame",
            "frames": FRAMES, "fps": 20, "duration_s": round(FRAMES / 20.0, 2),
            "panel_video": stats,
            "script_timeline": {"quiet": [0, 90], "left_press": [90, 160],
                                "right_burst_%d_N" % int(force(BURST_START + 17)[14]): [BURST_START,
                                                                                        BURST_END],
                                "emergency_stop_at": BURST_END, "release": [BURST_END, FRAMES]},
            "contact_tick_threshold_n": CONTACT_ON,
            "chain_provenance": {
                "burst_frames_[%d,%d)" % (BURST_START, BURST_END):
                    "seed 17, chunk 26 row 31, every stage logged end to end",
                "other frames": "seed 11, per-dimension |max| over the episode (never saturated)",
                "44.37": ("not shown here on purpose: section 1.1 establishes it is a post-"
                          "denormalisation magnitude in dims 0-5, so the old sample printing it in "
                          "the normalised row was wrong"),
                "units": ("model |max| per dim in normalised units, denorm/pm1 rows in LIBERO action "
                          "units, executed row in the +-1 rate command env.step receives"),
            },
            "stills": still,
            "sample_last_force_n": dict(l_in=float(force(FRAMES - 1)[12]),
                                        l_out=float(force(FRAMES - 1)[13]),
                                        r_in=float(force(FRAMES - 1)[14]),
                                        r_out=float(force(FRAMES - 1)[15])),
            "ok": not problems, "problems": problems}
    (OUT / "LAYOUT_SAMPLE_meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True))
    print(json.dumps({k: meta[k] for k in ("ok", "problems", "frames", "duration_s")}, sort_keys=True))
    print("video: %s  %d B  %s  %s" % (path, path.stat().st_size, stats["wh"], stats["backend"]))
    for s in still:
        print("still: %s" % s["png"])
    print("note : %s" % stats["notes"][-1])
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
