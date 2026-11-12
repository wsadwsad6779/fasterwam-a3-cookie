"""CPU proof that the composed recording is what it claims to be -- no GPU, no torch, no model.

The recording the user asked for has three properties that are each easy to lose silently: the two
camera images must survive untouched inside one MP4, the fingertip forces must be separate channels
read from the simulator, and the force curve must describe the frame it is drawn next to. None of
that can be checked by watching the video, and none of it can be checked on the GPU after 25 minutes
of inference, so it is checked here on synthetic data first.

Run:

    py -3.13 panelvideo_selftest40k.py        # local, cv2/mp4v
    python3 panelvideo_selftest40k.py         # pod, imageio/libx264
"""
from __future__ import annotations

import re
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import panelvideo40k as P  # noqa: E402

RESULTS = []
CH = P.PAD_ORDER


def check(name, ok, detail=""):
    RESULTS.append(bool(ok))
    print("%-68s %s  %s" % (name, "PASS" if ok else "FAIL", detail), flush=True)


def _raises(fn):
    try:
        fn()
    except Exception as exc:
        return "%s: %s" % (type(exc).__name__, str(exc)[:90])
    return None


def synth(n, seed=0):
    """A fake episode: four pad channels that come into contact at known frames."""
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float64)
    return {"l_inner": 300.0 * np.tanh((t - 120.0) / 25.0) * (t > 100.0) + 2.0 * rng.standard_normal(n),
            "l_outer": 180.0 * np.tanh((t - 132.0) / 25.0) * (t > 100.0) + 2.0 * rng.standard_normal(n),
            "r_inner": -220.0 * np.tanh((t - 300.0) / 30.0) * (t > 280.0) + 2.0 * rng.standard_normal(n),
            "r_outer": -95.0 * np.tanh((t - 315.0) / 30.0) * (t > 280.0) + 2.0 * rng.standard_normal(n)}


def ann_for(i, pads, n=600, **over):
    ann = {"frame": i, "total_frames": n, "seed": 17, "step": "step_040000", "ctrl_step": i,
           "chunk": i // 10,
           "pad_series": {c: pads[c][:i + 1] for c in CH},
           "pad_n": {c: float(pads[c][i]) for c in CH},
           "pad_l_tared_n": {"l_inner": float(pads["l_inner"][i]) - 1.4,
                             "l_outer": float(pads["l_outer"][i]) - 0.9},
           "wrist_n": [8.1, -2.3, 11.0], "wrist_torque_nm": [0.31, -0.08, 1.24],
           "gripper": "closed(-1)", "success": False, "terminated": False, "safety_reason": None,
           "model_absmax_dims": [33.5, 27.5, 21.5, 24.125, 18.5, 15.875, 40.0],
           "model_over5_frac": [0.523, 0.441, 0.491, 0.451, 0.440, 0.072, 0.617],
           # the four later stages are one row of one chunk, signed, as the real recorder stores them:
           # the seed 17 worst-row witness in ROOT_CAUSE_40K_20261112.md section 1.1
           "denorm_row": [1.3268, 0.0580, -4.3467, 0.6686, -0.3432, 0.0595, 40.0],
           "pm1_row": [1.3268, 0.0580, -4.3467, 0.6686, -0.3432, 0.0595, 1.0],
           "executed_row": [1.0, 0.116, -1.0, 1.0, 1.0, 1.0, -1.0, 0, 0, 0, 0, 0, 0, 1.0],
           "ik_cmd_row": [1.0, 0.116, -1.0, 1.0, 1.0, 1.0, -1.0, 0, 0, 0, 0, 0, 0, 1.0],
           "ticks": [{"frame": 100, "text": "L first contact"},
                     {"frame": 280, "text": "R first contact"}],
           "notes": ["env.step total 1.80 s per control step",
                     "force = raw env newtons, no zero-load tare (right pad has none)"]}
    ann.update(over)
    return ann


def cams(i):
    """Solid, index-coded camera images at the env's native size: any change to a camera pixel is
    then unmistakable."""
    f = np.zeros((P.CAM_SRC, P.CAM_SRC, 3), np.uint8)
    f[:] = (i % 200 + 8, (i * 3) % 180 + 8, 240 - (i % 200))
    w = np.zeros((P.CAM_SRC, P.CAM_SRC, 3), np.uint8)
    w[:] = (200 - (i % 190), 30 + (i % 150), (i * 7) % 240 + 8)
    return f, w


def col_mask(img, color, tol=12):
    return (np.abs(img.astype(np.int16) - np.asarray(color, np.int16)).max(axis=2) <= tol)


def playhead_columns(img, box, y0=None, y1=None):
    """Columns carrying the yellow playhead.

    Detection is by yellow dominance rather than an exact RGB match: the marker is 2 px wide and
    yuv420 encodes chroma at half resolution in both directions, so a decoded playhead is smeared
    over several columns and its colour is off by tens of levels.
    """
    x0, by0, x1, by1 = box
    y0 = by0 if y0 is None else y0
    y1 = by1 if y1 is None else y1
    reg = img[y0:y1, x0:x1].astype(np.int16)
    r, g, b = reg[..., 0], reg[..., 1], reg[..., 2]
    m = (r > 130) & (g > 130) & (np.abs(r - g) < 45) & ((r - b) > 60)
    if m.size == 0:
        return np.array([], dtype=int)
    cols = np.nonzero(m.sum(axis=0) >= 0.30 * m.shape[0])[0]
    return cols + x0


def expect_x(key, frame, boxes, total):
    """Recompute where frame `frame` must be drawn, without calling the composer's own helper."""
    x0, _y0, x1, _y1 = boxes[key]
    lo = 0 if key == "full" else max(0, frame - P.ZOOM_STEPS + 1)
    span = (total - 1) if key == "full" else (P.ZOOM_STEPS - 1)
    frac = (frame - lo) / float(span)
    return int(round((x0 + 1) + ((x1 - 1) - (x0 + 1)) * frac))


def main():
    out = Path(tempfile.mkdtemp(prefix="panelvideo_selftest_"))
    n = 600
    pads = synth(n)
    boxes = P.panel_boxes()

    # --- which sensor is which: the headline numbers are only worth anything if the columns are
    f18 = np.arange(P.FORCE18_W, dtype=np.float64)
    got, wrist, tau = P.force18_pads(f18)
    check("the four headline channels are FORCE columns 12/13/14/15, in that order",
          got == {"l_inner": 12.0, "l_outer": 13.0, "r_inner": 14.0, "r_outer": 15.0}
          and list(wrist) == [0.0, 1.0, 2.0] and list(tau) == [3.0, 4.0, 5.0],
          str(got) + " wrist=%s" % list(wrist))
    check("refuses a force vector that is not the env's 18-wide one (e.g. the 6 frozen channels)",
          _raises(lambda: P.force18_pads(np.zeros(6))) is not None
          and _raises(lambda: P.force18_pads(np.r_[np.zeros(17), np.nan])) is not None,
          _raises(lambda: P.force18_pads(np.zeros(6))))
    check("left and right are separate sides, and each side is inner | outer",
          P.FINGERTIP == {"L": ("l_inner", "l_outer"), "R": ("r_inner", "r_outer")},
          str(P.FINGERTIP))

    f0, w0 = cams(0)
    img, records = P.compose_frame(f0, w0, ann_for(0, pads))
    check("composed frame is one RGB image of the two cameras plus the panel",
          img.shape == (P.PANEL_H, 2 * P.CAM_W + P.PANEL_W, 3) and img.dtype == np.uint8,
          str(img.shape))
    cy = (P.PANEL_H - P.CAM_H) // 2
    front_px = img[cy:cy + P.CAM_H, 0:P.CAM_W]
    wrist_px = img[cy:cy + P.CAM_H, P.CAM_W:2 * P.CAM_W]
    check("each camera is shown at the historical size: the native frame enlarged 2x",
          P.CAM_W == P.CAM_SRC * P.CAM_SCALE == 512 and P.CAM_H == 512 and P.PANEL_H == 576,
          "camera %dx%d from %d at scale %d, canvas height %d"
          % (P.CAM_W, P.CAM_H, P.CAM_SRC, P.CAM_SCALE, P.PANEL_H))
    check("the front camera is a pure enlargement -- every 2x2 block is one source pixel, so "
          "nothing is drawn over it",
          np.array_equal(front_px[0::2, 0::2], f0)
          and front_px[1::2, ::2].tolist() == front_px[0::2, ::2].tolist()
          and front_px[:, 1::2].tolist() == front_px[:, 0::2].tolist(),
          "maxdiff=%d" % int(np.abs(front_px[0::2, 0::2].astype(int) - f0.astype(int)).max()))
    check("the left_wrist camera pixels are recovered verbatim the same way",
          np.array_equal(wrist_px[0::2, 0::2], w0),
          "maxdiff=%d" % int(np.abs(wrist_px[0::2, 0::2].astype(int)
                                    - w0.astype(int)).max()))
    check("the front camera is left of the left_wrist camera",
          boxes["front"][0] < boxes["left_wrist"][0] < boxes["panel"][0],
          "front=%s wrist=%s panel=%s" % (boxes["front"], boxes["left_wrist"], boxes["panel"]))
    check("ink never crosses into a camera column",
          all(r[1] >= 2 * P.CAM_W for r in records),
          "min record x=%d" % min(r[1] for r in records))
    check("every text record stays inside the panel's own rectangle",
          all(boxes["panel"][0] <= r[1] and r[3] <= boxes["panel"][2]
              and 0 <= r[2] and r[4] <= P.PANEL_H for r in records),
          "max x1=%d, max y1=%d, panel right=%d" %
          (max(r[3] for r in records), max(r[4] for r in records), boxes["panel"][2]))
    check("the text block ends above the first curve box (no overlap, no clipping)",
          max(r[4] for r in records) <= boxes["full"][1],
          "last text y=%d, curve top=%d" % (max(r[4] for r in records), boxes["full"][1]))
    late = 400
    img_late, _recs_late = P.compose_frame(*cams(late), ann_for(late, pads))
    zx0, zy0, zx1, zy1 = boxes["zoom"]
    counts = {}
    for c in CH:
        reg = img_late[zy0:zy1, zx0:zx1].astype(np.int16)
        counts[c] = int((np.abs(reg - np.asarray(P.PAD_COLORS[c], np.int16)).max(axis=2)
                         <= 8).sum())
    check("all four pad channels are drawn as their own curve, in four distinct colours",
          all(v > 100 for v in counts.values()) and len(set(P.PAD_COLORS.values())) == 4,
          "ink px in the zoom box=%s" % counts)
    check("no ink is drawn past the panel's right edge (a label wider than its column)",
          int((np.abs(img[boxes["text"][1]:boxes["text"][3],
                       boxes["panel"][2] - 6:boxes["panel"][2]]).astype(int)
              - np.asarray(P.BG, int)).max()) <= 24,
          "right margin columns %d..%d" % (boxes["panel"][2] - 6, boxes["panel"][2]))
    wide = ann_for(0, pads)
    wide["notes"] = ["x" * 200]
    _img_wide, recs_wide = P.compose_frame(*cams(0), wide)
    long_rec = [r for r in recs_wide if r[0].startswith("xxx")]
    check("a note wider than the panel is cut with a visible ellipsis, not run off the edge",
          len(long_rec) == 1 and long_rec[0][0].endswith("...")
          and long_rec[0][3] <= boxes["text"][2],
          "record x=%d..%d of text box %d..%d, tail=%r"
          % (long_rec[0][1], long_rec[0][3], boxes["text"][0], boxes["text"][2],
             long_rec[0][0][-8:]) if long_rec else "no long note drawn at all")
    # --- the action chain: four stages, four different units, each line named by the variable it reads
    STAGES = ("model out |max| per dim", "model out rows |x|>5", "denormalised row",
              "pm1 gripper col", "env.step cmd", "at the +-1 rail")
    drawn = {p: [r[0] for r in records if r[0].startswith(p)] for p in STAGES}
    missing = [p for p, hits in drawn.items() if len(hits) != 1]
    cut = [p for p, hits in drawn.items() if hits and hits[0].endswith("...")]
    check("the chain draws exactly one line per stage, six lines, each naming its own units",
          not missing, "missing or doubled: %s" % missing)
    check("no stage line is wide enough to need an ellipsis, so no number is cut off",
          not cut, "cut lines: %s" % cut)
    dn = [float(x) for x in re.findall(r"[-+]\d+\.\d+", drawn["denormalised row"][0])]
    ex = [float(x) for x in re.findall(r"[-+]\d+\.\d+", drawn["env.step cmd"][0])]
    check("the executed line is the +-1 rate command; the stage before it is allowed out of that band",
          len(dn) == 7 and len(ex) == 7 and max(abs(v) for v in ex) <= 1.0 + 1e-9
          and max(abs(v) for v in dn) > 1.0,
          "cmd absmax=%.3f vs denormalised absmax=%.3f"
          % (max(abs(v) for v in ex), max(abs(v) for v in dn)))
    want_sat = int(np.sum(np.abs(np.asarray(ann_for(0, pads)["executed_row"][:7])) >= 1.0 - 1e-6))
    check("the rail count and the IK-equality note are recomputed from those rows",
          drawn["at the +-1 rail"][0].startswith("at the +-1 rail: %d/7" % want_sat)
          and "IK input == cmd: yes" in drawn["at the +-1 rail"][0],
          drawn["at the +-1 rail"][0])
    noact = ann_for(0, pads)
    for k in ("executed_row", "ik_cmd_row", "model_absmax_dims", "model_over5_frac",
              "denorm_row", "pm1_row"):
        noact.pop(k, None)
    _img_na, recs_na = P.compose_frame(*cams(0), noact)
    check("a frame that recorded no action draws no action line, which is frame 0's case",
          not any(r[0].startswith(s) for r in recs_na for s in STAGES),
          str([r[0] for r in recs_na if r[0].startswith(("model", "denorm", "pm1", "env", "at th"))]))

    check("a left curve is never mistaken for a right one (hue families separable by >=60)",
          min(abs(int(P.PAD_COLORS[a][0]) - int(P.PAD_COLORS[b][0]))
              for a in P.FINGERTIP["L"] for b in P.FINGERTIP["R"]) >= 60,
          "L=%s R=%s" % ([P.PAD_COLORS[c] for c in P.FINGERTIP["L"]],
                         [P.PAD_COLORS[c] for c in P.FINGERTIP["R"]]))
    big = [r for r in records if r[0].endswith("_value")]
    others = [r for r in records if not r[0].endswith("_value")]
    check("the four fingertip numbers are the largest text in the panel",
          len(big) == 4 and min(r[4] - r[2] for r in big) > 25
          and max(r[4] - r[2] for r in others) < 20,
          "pad glyph h=%s, next tallest=%d" % ([r[4] - r[2] for r in big],
                                               max(r[4] - r[2] for r in others)))
    check("the L and R number pairs do not overlap each other or the panel edge",
          max(r[3] for r in big if r[0].startswith("L_pad"))
          <= min(r[1] for r in big if r[0].startswith("R_pad"))
          and max(r[3] for r in big) <= boxes["panel"][2],
          "L right edge=%d, R left edge=%d" %
          (max(r[3] for r in big if r[0].startswith("L_pad")),
           min(r[1] for r in big if r[0].startswith("R_pad"))))

    # --- synchronisation, at the pixel level, before any encoder is involved
    k = 137
    fr, fr_recs = P.compose_frame(*cams(k), ann_for(k, pads))
    nrange = P.nice_range(*[pads[c][:k + 1] for c in CH])
    for key in ("full", "zoom"):
        want = expect_x(key, k, boxes, n)
        cols = playhead_columns(fr, boxes[key])
        check("playhead sits on frame %d in the %s curve" % (k, key),
              len(cols) >= 1 and abs(int(np.median(cols)) - want) <= 2,
              "detected=%s expected=%d" % (list(cols[:4]), want))
    mid = 300
    fr_mid, _r = P.compose_frame(*cams(mid), ann_for(mid, pads))
    for key in ("full", "zoom"):
        want = expect_x(key, mid, boxes, n)
        cols = playhead_columns(fr_mid, boxes[key])
        check("the overview playhead sweeps: frame %d of %d is inside the %s box" % (mid, n, key),
              len(cols) >= 1 and abs(int(np.median(cols)) - want) <= 2
              and (boxes[key][0] + 1 < want < boxes[key][2] - 2 or key == "zoom"),
              "detected=%s expected=%d box=%s" % (list(cols[:4]), want, boxes[key]))
    check("the overview playhead is near the middle of the box at half the episode",
          abs(expect_x("full", mid, boxes, n)
              - 0.5 * (boxes["full"][0] + boxes["full"][2])) < 12,
          "x=%d, box centre=%d" % (expect_x("full", mid, boxes, n),
                                   int(0.5 * (boxes["full"][0] + boxes["full"][2]))))
    px = expect_x("full", k, boxes, n)
    for c in CH:
        m = col_mask(fr, P.PAD_COLORS[c], tol=24)
        rows = np.nonzero(m[boxes["full"][1]:boxes["full"][3], px])[0]
        want_y = P.curve_y(pads[c][k], nrange, boxes["full"][1] + 14,
                           boxes["full"][3] - 2) - boxes["full"][1]
        check("the %s marker is at the pixel its own newton value maps to" % c,
              len(rows) > 0 and abs(int(np.median(rows)) - want_y) <= 4,
              "marker rows=%s expected offset=%d" % (list(rows[:6]), want_y))

    # a change at frame k must not touch any earlier frame, and must touch frame k
    pads2 = {c: pads[c].copy() for c in CH}
    pads2["l_inner"][k] = pads["l_inner"][k] + 90.0
    same_before = all(np.array_equal(P.compose_frame(*cams(i), ann_for(i, pads2))[0],
                                     P.compose_frame(*cams(i), ann_for(i, pads))[0])
                      for i in (0, k // 2, k - 1))
    diff_at_k = P.compose_frame(*cams(k), ann_for(k, pads2))[0] != \
        P.compose_frame(*cams(k), ann_for(k, pads))[0]
    check("editing the trace at frame k leaves frames < k bit-identical", same_before,
          "checked frames 0..%d" % (k - 1))
    check("editing the trace at frame k changes frame k's panel", bool(diff_at_k.any()),
          "changed px=%d" % int(diff_at_k.sum()))
    img_k_pads2, recs_pads2 = P.compose_frame(*cams(k), ann_for(k, pads2))
    pads_r = {c: (pads[c] + 50.0 if c == "r_inner" else pads[c]) for c in CH}
    img_k_rchanged, recs_rchanged = P.compose_frame(*cams(k), ann_for(k, pads_r))
    check("a change to the right pad at frame k does not move the left pad's headline",
          np.array_equal(_l_glyph(img_k_rchanged, recs_rchanged), _l_glyph(fr, fr_recs)),
          "glyph px sum L=%s/%s" % (int(_l_glyph(img_k_rchanged, recs_rchanged).sum()),
                                    int(_l_glyph(fr, fr_recs).sum())))
    check("a change to the left pad at frame k does move it (the check above is not vacuous)",
          not np.array_equal(_l_glyph(img_k_pads2, recs_pads2), _l_glyph(fr, fr_recs)),
          "glyph px sum L=%s/%s" % (int(_l_glyph(img_k_pads2, recs_pads2).sum()),
                                    int(_l_glyph(fr, fr_recs).sum())))
    check("curve_x / curve_y are monotone and pinned to the box edges",
          P.curve_x(0, n, 10, 90) == 10 and P.curve_x(n - 1, n, 10, 90) == 90
          and all(P.curve_x(i + 1, n, 10, 90) >= P.curve_x(i, n, 10, 90) for i in range(n - 1))
          and P.curve_y(0.0, 100.0, 0, 100) == 50 and P.curve_y(100.0, 100.0, 0, 100) < 10,
          "x0=%d xN=%d y(0)=%d y(max)=%d" % (P.curve_x(0, n, 10, 90), P.curve_x(n - 1, n, 10, 90),
                                             P.curve_y(0.0, 100.0, 0, 100),
                                             P.curve_y(100.0, 100.0, 0, 100)))

    # --- the interlocks that keep a mis-wired recording from producing a plausible-looking video
    def _guard(tag, mutate, frame=1):
        """Build a mis-wired annotation and assert the composer refuses it.

        `rec.n` is set directly so that every case except the ordering case starts from a state
        where the frame index is legal; otherwise all of the guards would "pass" by tripping the
        first check rather than the one under test.
        """
        rec = P.PanelRecorder(out / ("interlock_%s.mp4" % tag))
        rec.n = frame + 1 if tag == "order" else frame
        ann = ann_for(frame, pads, n=(0 if tag == "noaxis" else 600))
        mutate(ann)
        rec.add(*cams(frame), ann)

    def _short(a):
        a["pad_series"] = {c: a["pad_series"][c][:a["frame"]] for c in CH}

    def _nan(a):
        bad = a["pad_series"]["r_outer"].copy()
        bad[-1] = np.nan
        a["pad_series"]["r_outer"] = bad

    def _stale(a):
        a["pad_n"] = dict(a["pad_n"], r_inner=float(a["pad_n"]["r_inner"]) + 1.0)

    def _missing(a):
        a["pad_series"] = {c: v for c, v in a["pad_series"].items() if c != "r_outer"}

    def _ragged(a):
        a["pad_series"] = dict(a["pad_series"], l_outer=a["pad_series"]["l_outer"][:-1])

    bad = [
        ("video frames out of order", lambda a: None, "order"),
        ("a force trace shorter than the frame index", _short, "short"),
        ("NaN in one pad trace", _nan, "nan"),
        ("the number shown is not the last sample of its own trace", _stale, "stale"),
        ("a fingertip channel missing from the trace (the right pad is the tempting one)",
         _missing, "missing"),
        ("the four traces disagree about their length", _ragged, "ragged"),
        ("a camera frame resized", None, "resize"),
        ("the curve axis is not pinned to the episode length",
         lambda a: a.pop("total_frames"), "noaxis"),
        ("panel text would overflow into the curves",
         lambda a: a.update(notes=["overflow line %02d" % i for i in range(40)]), "overflow"),
    ]
    for name, mutate, tag in bad:
        if tag == "resize":
            msg = _raises(lambda: P.compose_frame(np.zeros((128, 128, 3), np.uint8), w0,
                                                   ann_for(0, pads)))
        else:
            msg = _raises(lambda m=mutate, k=tag: _guard(k, m))
        check("refuses to record: %s" % name, msg is not None, msg or "NO ERROR RAISED")

    # --- write it, read it back
    m_frames = 120
    rec = P.PanelRecorder(out / "seed17_step040000_composed.mp4", fps=20)
    for i in range(m_frames):
        rec.add(*cams(i), ann_for(i, pads, n=m_frames))
    stats = rec.close()
    check("mp4 written by %s, non-trivial size" % stats["backend"],
          stats["frames_written"] == m_frames and stats["bytes"] > 20000,
          "%d frames, %.1f KB, %s" % (stats["frames_written"], stats["bytes"] / 1024.0,
                                      stats["wh"]))
    back = P.read_frames(stats["path"])
    check("decodes back with the requested frame count", len(back) == m_frames,
          "%d of %d" % (len(back), m_frames))
    check("decoded geometry matches the composed layout",
          back and back[0].shape == (P.PANEL_H, 2 * P.CAM_W + P.PANEL_W, 3),
          str(None if not back else back[0].shape))
    check("both camera regions are the same width and sit side by side",
          boxes["left_wrist"][0] - boxes["front"][0] == P.CAM_W
          and boxes["panel"][0] - boxes["left_wrist"][0] == P.CAM_W,
          "%d px each: %s %s" % (P.CAM_W, boxes["front"], boxes["left_wrist"]))
    errs = []
    for i in (0, 17, 61, 103, m_frames - 1):
        fr_b = back[i][cy:cy + P.CAM_H, 0:P.CAM_W].astype(np.int16)[0::2, 0::2]
        errs.append(float(np.abs(fr_b - cams(i)[0].astype(np.int16)).mean()))
    check("frame i still carries camera i (no re-ordering, no drift)", max(errs) < 8.0,
          "per-frame mean abs error=%s" % [round(e, 2) for e in errs])
    play = []
    for i in (33, 99):
        cols = playhead_columns(back[i], boxes["zoom"],
                                boxes["zoom"][1] + 4, boxes["zoom"][3] - 4)
        want = expect_x("zoom", i, boxes, m_frames)
        play.append((i, int(np.median(cols)) if len(cols) else -1, want))
    check("the playhead survives encoding at the right frame",
          all(a >= 0 and abs(a - b) <= 5 for _i, a, b in play),
          "detected/expected=%s" % play)
    still_four = []
    for c in CH:
        reg = back[99][boxes["zoom"][1] + 2:boxes["zoom"][3] - 2,
                       boxes["zoom"][0]:boxes["zoom"][2]].astype(np.int16)
        want = np.asarray(P.PAD_COLORS[c], np.int16)
        # the encoder moves a 1-2 px line a long way, so this is existence, not colour fidelity
        near = (np.abs(reg - want).max(axis=2) <= 70).sum()
        still_four.append((c, int(near)))
    check("all four channel colours still appear in the decoded curve box",
          all(v >= 20 for _c, v in still_four), str(still_four))
    (out / "_last.txt").write_text(str(stats))
    print("\nartifacts: %s" % out)
    n_fail = RESULTS.count(False)
    print("%s: %d/%d checks passed" % ("FAIL" if n_fail else "PASS",
                                       len(RESULTS) - n_fail, len(RESULTS)))
    return 1 if n_fail else 0


def _l_glyph(img, records):
    """Ink mask of the left fingertip's two big numbers, cropped by their own text records."""
    big = [r for r in records if r[0] == "L_pad_value"]
    if len(big) != 2:
        raise AssertionError("expected two L_pad_value records, got %d" % len(big))
    x0 = min(r[1] for r in big)
    x1 = max(r[3] for r in big)
    y0 = min(r[2] for r in big)
    y1 = max(r[4] for r in big)
    reg = img[y0:y1, x0:x1].astype(np.int16)
    return (np.abs(reg - np.asarray(P.BG, np.int16)).max(axis=2) > 24)


if __name__ == "__main__":
    sys.exit(main())
