"""Compose a rollout into ONE mp4: [front | left_wrist | diagnostics panel].

The batch's own recorder (`record40k.py`) writes one mp4 *per camera* and burns the numbers into
each of them. That layout answers "what did the robot see" but not the question actually on the
table now, which is when each fingertip touched and how hard -- so this file builds the other
layout: the two camera images side by side, untouched, and every number, force curve and status
string in a separate panel to the right. Nothing is ever drawn over a camera image; that is not a
style rule but an invariant the composer enforces (each camera is a pure pixel-replication of the
env's own 256x256 frame, and the selftest recovers it and compares it pixel for pixel).

Two contracts, because both are ways this evidence could silently become meaningless:

* **The force panel shows simulator-measured newtons.** The four fingertip-pad channels are columns
  12/13 (left inner/outer) and 14/15 (right inner/outer) of the env's 18-wide `observation.force`
  vector -- the layout in `chan18_audit.py`'s NAMES, the same vector `force_stream.CHANNELS` gathers
  six channels out of. They are never `pred["force_future_newton"]`: the model's own future-force
  head is a *prediction*, and drawing it next to a frame as if it were the reading would make the
  panel unfalsifiable. `nefc`/`ncon` are constraint counts, not newtons, so they never appear on a
  newton axis. Nothing is summed across fingertips: an "L_pad" and an "R_pad" that had been added
  together would answer a question nobody asked.
* **The curve is frame-synchronous.** `pad_series[ch][i]` must be the force measured at the frame
  that produced video frame ``i``. `PanelRecorder.add` asserts len(series) == frame + 1 for all four
  channels, and that the number displayed is the last sample of its own trace, so an off-by-one --
  the classic way a force trace ends up describing the *next* contact -- fails at the first replan
  instead of yielding a video that looks fine.

Layout: the canvas is 576 px tall and the two cameras occupy the left two columns at 512x512 each,
which is the historical result video's own scale -- the env renders 256x256 and the delivered
annotated mp4 showed it enlarged 2x, so the enlargement is what "same style" means here. It is done
by pixel replication, not interpolation, so the camera is still recoverable verbatim (the selftest
takes every other row and column and compares against the source) and nothing is drawn on top of it.
The panel is centred vertically in its own columns, and the text block is ordered by priority --
fingertip forces first, then status, then the action chain. If a frame's content cannot fit, `_Stack`
raises rather than drawing over the curves or clipping a number, because an evidence video with a
silently dropped field is worse than a failed recording run.

Encoder note: on the pod this uses imageio/libx264 + yuv420p, which is what `record40k.py` and the
FasterWAM trainer both use (`fw/src/fasterwam/utils/video_io.py`), so the codec stays comparable
with the historical result videos. Where imageio is unavailable (a CPU-only workstation) it falls
back to OpenCV's mp4v writer so the layout and the synchronisation can still be verified without a
GPU; the backend actually used is reported in the stats, because "verified the layout" and
"verified the pod's bitstream" are different claims.
"""

from __future__ import annotations

import numpy as np

# Measured off the historical FasterWAM result video (`r4_c0_seed21_annotated_20fps.mp4`, the one
# file in the delivered package that already has a camera plus a diagnostics panel): 864x576, h264,
# 20 fps, 600 frames for a 600-step episode, and the camera block is the env's native 256x256 frame
# shown at 512x512 -- a 2x enlargement, at the same height as the whole canvas. The two-camera
# version the user asked for therefore keeps 576 as the canvas height and 512 as the on-screen
# camera size, and only grows wider, because a second camera column is what the new layout adds.
CAM_SRC = 256
CAM_SCALE = 2
CAM_W = CAM_SRC * CAM_SCALE
CAM_H = CAM_SRC * CAM_SCALE
PANEL_W = 640
PANEL_H = 576
ZOOM_STEPS = 120
TEXT_H = 300
CURVE_GAP = 10

BG = (16, 18, 22)
GRID = (66, 72, 82)
TXT = (225, 230, 235)
DIM = (150, 158, 168)
COL_L = (60, 255, 200)
COL_R = (255, 170, 40)
COL_PLAY = (255, 255, 0)
COL_WARN = (255, 70, 70)

# The fingertip pads. Each fingertip has two strain sensors, so "L_pad" and "R_pad" are two pairs,
# not two scalars, and the env's own channel table says which is which (`chan18_audit.py`'s NAMES,
# the same 18-wide layout `force_stream.CHANNELS` gathers six of). Pairing a value with a column
# number that no longer means what it did would relabel the headline of the whole deliverable, so
# the width is asserted on every frame rather than assumed.
FORCE18_W = 18
PAD_COLS = {"l_inner": 12, "l_outer": 13, "r_inner": 14, "r_outer": 15}
PAD_ORDER = ("l_inner", "l_outer", "r_inner", "r_outer")
FINGERTIP = {"L": ("l_inner", "l_outer"), "R": ("r_inner", "r_outer")}
PAD_COLORS = {"l_inner": COL_L, "l_outer": (26, 150, 118),
              "r_inner": COL_R, "r_outer": (158, 100, 18)}
PAD_TICKER = {"l_inner": "L in", "l_outer": "L out", "r_inner": "R in", "r_outer": "R out"}
WRIST_FORCE_COLS = (0, 1, 2)      # L_wrist_fx/fy/fz, newtons
WRIST_TORQUE_COLS = (3, 4, 5)     # L_wrist_tx/ty/tz, newton-metres

_MONO = ("C:/Windows/Fonts/consolab.ttf", "C:/Windows/Fonts/consola.ttf", "C:/Windows/Fonts/lucon.ttf",
         "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
         "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf")
_SANS = ("C:/Windows/Fonts/arial.ttf",
         "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")


def _font(px):
    from PIL import ImageFont

    for path in _MONO + _SANS:
        try:
            return ImageFont.truetype(path, px)
        except Exception:
            continue
    try:
        return ImageFont.load_default(px)
    except TypeError:
        return ImageFont.load_default()


class _Stack:
    """A text cursor for the panel. Draws (text, px, color, x, y) records and never overflows."""

    def __init__(self, draw, x, y_top, y_bottom, x_right=None):
        self.d = draw
        self.x = int(x)
        self.y = int(y_top)
        self.y_bottom = int(y_bottom)
        self.x_right = None if x_right is None else int(x_right)
        self.records = []

    def line(self, text, px=11, color=TXT, gap=3, x=None, font=None):
        font = _font(px) if font is None else font
        box = font.getbbox(str(text))
        h, w = box[3] - box[1], box[2] - box[0]
        if self.y + h > self.y_bottom:
            raise OverflowError("panel text block overflow: %r needs y=%d+h=%d > %d"
                                % (str(text)[:48], self.y, h, self.y_bottom))
        cx = self.x if x is None else int(x)
        if self.x_right is not None and cx + w > self.x_right:
            text = self._fit(text, font, self.x_right - cx)
            w = font.getbbox(str(text))[2] - font.getbbox(str(text))[0]
        self.d.text((cx, self.y), str(text), font=font, fill=color)
        self.records.append((str(text), cx, self.y, cx + w, self.y + h))
        self.y += h + gap
        return self.records[-1]

    def space(self, px):
        self.y += int(px)

    @staticmethod
    def _fit(text, font, max_w):
        """Cut a line to the box instead of letting it run under the panel edge.

        The cut is visible -- an ellipsis on the same line -- because a silently clipped note is how
        a calibration caveat stops existing in the delivered video.
        """
        out = str(text)
        while out and font.getbbox(out + "...")[2] > max_w:
            out = out[:-1]
        return out + "..." if out != str(text) else str(text)

    @property
    def bounds(self):
        return (self.x, self.y, self.y_bottom)


def _num(v, width, prec):
    if v is None:
        return "n/a".rjust(width)
    v = float(v)
    if not np.isfinite(v):
        return "n/a".rjust(width)
    return (("%+." + str(prec) + "f") % v).rjust(width)


def _row(vals, prec=2):
    width = 6 + prec
    return " ".join(_num(v, width, prec) for v in vals)


def nice_range(*series, floor=20.0):
    """A symmetric newton axis big enough for every series given, rounded to a readable step."""
    m = 0.0
    for s in series:
        a = np.abs(np.asarray(s, dtype=np.float64))
        if a.size:
            m = max(m, float(np.nanmax(a)))
    if not np.isfinite(m) or m <= 0.0:
        m = floor
    m = max(float(m), floor) * 1.08
    for step in (1.0, 2.0, 5.0, 10.0, 20.0, 25.0, 50.0, 100.0, 200.0, 250.0, 500.0, 1000.0, 2000.0):
        if m <= step:
            return step
    return 10.0 ** int(np.ceil(np.log10(m)))


def curve_x(i, n, x0, x1):
    """Pixel column of sample i of n, with sample 0 pinned to x0 and sample n-1 to x1."""
    n = int(n)
    if n <= 1:
        return int(x0)
    return int(round(x0 + (x1 - x0) * (i / (n - 1.0))))


def curve_y(value, nrange, y_top, y_bottom):
    """Pixel row of a newton value: +nrange at y_top, 0 in the middle, -nrange at y_bottom."""
    mid = 0.5 * (y_top + y_bottom)
    half = 0.5 * (y_bottom - y_top)
    frac = max(-1.0, min(1.0, float(value) / float(nrange)))
    return int(round(mid - frac * half))


def panel_boxes(text_h=TEXT_H, pad=6):
    """Fixed geometry, in composed-frame pixels: camera columns, then the panel's three regions."""
    cy0 = (PANEL_H - CAM_H) // 2
    x0, x1 = 2 * CAM_W + 6, 2 * CAM_W + PANEL_W - 6
    rest0 = pad + text_h + CURVE_GAP
    rest1 = PANEL_H - pad
    avail = rest1 - rest0 - CURVE_GAP
    a_h = int(avail * 0.5)
    return {"front": (0, cy0, CAM_W, cy0 + CAM_H),
            "left_wrist": (CAM_W, cy0, 2 * CAM_W, cy0 + CAM_H),
            "panel": (2 * CAM_W, 0, 2 * CAM_W + PANEL_W, PANEL_H),
            "text": (x0, pad, x1, pad + text_h),
            "full": (x0, rest0, x1, rest0 + a_h),
            "zoom": (x0, rest0 + a_h + CURVE_GAP, x1, rest1)}


def pad_dict(ann, key):
    """The four fingertip-pad channels of one annotation, as arrays (key="pad_series") or floats.

    Both the trace and the headline number go through here, so a recording that leaves out a channel
    -- the right pad especially, which is not one of the six channels the model conditions on and so
    has no other reason to exist in the wiring -- fails at the first frame instead of quietly drawing
    two curves.
    """
    src = ann.get(key)
    if not isinstance(src, dict) or set(src) != set(PAD_ORDER):
        raise ValueError("%s must carry exactly the four fingertip channels %s, got %r"
                         % (key, list(PAD_ORDER), None if src is None else sorted(src)))
    out = {}
    for ch in PAD_ORDER:
        v = src[ch]
        if key == "pad_series":
            arr = np.asarray(v, dtype=np.float64).reshape(-1)
            if arr.ndim != 1 or arr.size == 0:
                raise ValueError("pad_series[%r] must be a non-empty 1-D trace" % ch)
            out[ch] = arr
        else:
            out[ch] = float(v)
    return out


def force18_pads(force18):
    """Pull the four pad channels (and the left wrist force/torque) out of the env's raw vector.

    This is the only place a column number is trusted, so it is the only place a width assertion is
    needed: an env that changed the vector's layout would otherwise silently relabel the headline
    numbers of the deliverable.
    """
    f = np.asarray(force18, dtype=np.float64).reshape(-1)
    if f.size != FORCE18_W:
        raise ValueError("the env's force vector is %d wide, expected %d; the pad channel indices "
                         "below would then name the wrong sensor" % (f.size, FORCE18_W))
    if not np.isfinite(f).all():
        raise ValueError("the env's force vector has a NaN/inf; refusing to draw it as newtons")
    return ({"l_inner": float(f[PAD_COLS["l_inner"]]), "l_outer": float(f[PAD_COLS["l_outer"]]),
             "r_inner": float(f[PAD_COLS["r_inner"]]), "r_outer": float(f[PAD_COLS["r_outer"]])},
            np.array([f[c] for c in WRIST_FORCE_COLS], dtype=np.float64),
            np.array([f[c] for c in WRIST_TORQUE_COLS], dtype=np.float64))


def draw_text_block(st, ann, boxes):
    """Everything numeric, in priority order: fingertip forces, status, then the action chain."""
    st.line("seed %-4s  %s  front | left_wrist" % (ann.get("seed", "?"), ann.get("step", "?")),
            13, TXT)
    st.line("frame %04d/%-4s  ctrl %-4s  chunk %s" %
            (int(ann["frame"]), ann.get("total_frames", "?"), ann.get("ctrl_step", "?"),
             ann.get("chunk", "?")), 11, TXT)
    st.space(4)

    x0, _y0, x1, _y1 = boxes["text"]
    half = (x1 - x0) // 2
    pn = pad_dict(ann, "pad_n")
    big = _font(30)
    lab = _font(13)
    row_y = st.y
    labels = (("L_pad_label", x0, "L_pad  left fingertip (N)  in | out", COL_L),
              ("R_pad_label", x0 + half, "R_pad  right fingertip (N)  in | out", COL_R))
    for tag, col, text, color in labels:
        st.d.text((col, row_y), text, font=lab, fill=color)
        w = lab.getbbox(text)[2]
        st.records.append((tag, col, row_y, col + w, row_y + 15))
    for side, cols in (("L_pad", FINGERTIP["L"]), ("R_pad", FINGERTIP["R"])):
        col = x0 if side == "L_pad" else x0 + half
        sub = half // 2 - 8
        for j, ch in enumerate(cols):
            txt = _num(pn[ch], 7, 1).strip()
            st.d.text((col + j * (sub + 8), row_y + 17), txt, font=big,
                      fill=PAD_COLORS[ch])
            st.records.append(("%s_value" % side, col + j * (sub + 8), row_y + 17,
                               col + j * (sub + 8) + sub, row_y + 17 + 36))
    st.y = row_y + 58
    wrist = ann.get("wrist_n")
    if wrist is not None:
        w = np.asarray(wrist, dtype=np.float64)
        line = "wrist L fx/fy/fz %s  |F| %s N" % (_row(w, 1), _num(np.linalg.norm(w), 5, 1))
        tq = ann.get("wrist_torque_nm")
        if tq is not None:
            line += "  tau %s Nm" % _row(np.asarray(tq, dtype=np.float64), 2)
        st.line(line, 11, TXT)
    tare = ann.get("pad_l_tared_n")
    if tare is not None:
        st.line("L inner/outer after the model's zero-load tare %s N"
                % _row([tare["l_inner"], tare["l_outer"]], 1), 11, DIM)
    st.space(3)

    reason = ann.get("safety_reason")
    st.line("gripper %-8s  success %-5s  terminated %-5s" %
            (str(ann.get("gripper", "?")), str(ann.get("success", "?")),
             str(bool(ann.get("terminated", False)))), 11, TXT)
    st.line("safety_reason: %s" % ("none" if reason in (None, "") else str(reason)[:60]),
            11, COL_WARN if reason else DIM)

    md = ann.get("model_absmax_dims")
    if md is not None:
        st.line("model out |max| per dim [norm units] %s" % _row(md, 2), 11, TXT)
    ov = ann.get("model_over5_frac")
    if ov is not None:
        st.line("model out rows |x|>5 [frac]  %s" % " ".join("%5.3f" % float(v) for v in ov),
                11, COL_WARN if float(np.max(ov)) > 0 else DIM)
    den = ann.get("denorm_row")
    if den is not None:
        st.line("denormalised row [action units] %s" % _row(den, 3), 11, TXT)
    pm1 = ann.get("pm1_row")
    if pm1 is not None:
        st.line("pm1 gripper col [action units] %s" % _row(pm1, 3), 11, TXT)
    ex = ann.get("executed_row")
    if ex is not None:
        ex = np.asarray(ex, dtype=np.float64).reshape(-1)
        st.line("env.step cmd L0-6 [rate +-1] %s" % _row(ex[:7], 3), 11, TXT)
        sat = int(np.sum(np.abs(ex[:7]) >= 1.0 - 1e-6))
        # Counted on the post-clip row, so the honest claim is "sitting exactly on the rail", not
        # "this value was clipped": an adapter output that happened to be 1.0 already looks the same.
        line = "at the +-1 rail: %d/7 (adapter v3 clips silently) " % sat
        ik = ann.get("ik_cmd_row")
        if ik is not None:
            ik = np.asarray(ik, dtype=np.float64).reshape(-1)
            n = min(ex.size, ik.size)
            diff = float(np.abs(ex[:n] - ik[:n]).max()) if n else 0.0
            line += " | IK input == cmd: %s" % ("yes" if diff == 0.0 else "NO %.4f" % diff)
        st.line(line, 11, COL_WARN if sat else DIM)
    for note in ann.get("notes", []):
        st.line(str(note), 11, DIM)


def episode_axis(ann):
    """The overview's horizontal axis is pinned to the episode's control-step budget.

    Stretching it to however many frames have been recorded so far would keep the playhead parked
    at the right edge of the box for the whole video, which destroys the one thing the position is
    for: showing how far into the episode this frame is.
    """
    frame = int(ann["frame"])
    total = ann.get("total_frames")
    try:
        total = int(total)
    except (TypeError, ValueError):
        raise ValueError("total_frames must be the number of control steps the episode can run "
                         "(it pins the force-curve axis); got %r" % (total,))
    if total <= frame:
        raise ValueError("total_frames=%d cannot contain frame %d" % (total, frame))
    return total


def draw_curve(d, box, lo, hi, axis_n, nrange, series, ticks, frame, title=""):
    """One newton-vs-control-step plot.

    `axis_n` is the number of control steps the horizontal axis spans -- the episode length for the
    overview, `ZOOM_STEPS` for the trailing window. Pinning it (instead of stretching whatever has
    been recorded so far) is what makes the playhead's column mean something: it sweeps across the
    overview as the episode runs, and a viewer can read "how far into the 600 steps is this frame"
    from its position alone.

    `series` is a list of (name, array, colour); every fingertip channel is its own polyline, so no
    two sensors are ever summed into one curve.
    """
    x0, y0, x1, y1 = box
    d.rectangle(box, fill=BG, outline=GRID)
    mid = (y0 + y1) // 2
    d.line([(x0, mid), (x1, mid)], fill=GRID, width=1)
    n = max(int(axis_n), 1)
    d.text((x0 + 3, y0 + 2), "%s   axis +-%g N" % (title, nrange), font=_font(11), fill=DIM)
    for yy, lab in ((y0 + 14, "+%g" % nrange), (mid - 5, "0"), (y1 - 13, "-%g" % nrange)):
        d.text((x1 - 58, yy), lab, font=_font(11), fill=DIM)
    xs = [curve_x(i - lo, n, x0 + 1, x1 - 1) for i in range(lo, min(hi, len(series[0][1])))]
    if len(xs) >= 2:
        for _name, arr, color in series:
            ys = [curve_y(arr[i], nrange, y0 + 14, y1 - 2)
                  for i in range(lo, min(hi, len(arr)))]
            d.line(list(zip(xs, ys)), fill=color, width=1)
    for f, _text in ticks:
        if lo <= f < lo + n:
            tx = curve_x(f - lo, n, x0 + 1, x1 - 1)
            d.line([(tx, y0 + 1), (tx, y1 - 1)], fill=COL_WARN, width=1)
    if lo <= frame < lo + n and len(series[0][1]):
        px = curve_x(frame - lo, n, x0 + 1, x1 - 1)
        d.line([(px, y0 + 1), (px, y1 - 1)], fill=COL_PLAY, width=2)
        for _name, arr, color in series:
            if frame < len(arr):
                yy = curve_y(arr[frame], nrange, y0 + 14, y1 - 2)
                d.ellipse([px - 3, yy - 3, px + 3, yy + 3], outline=color, width=2)
        nx, ny = px + 4, y1 - 16
        if nx > x1 - 42:                 # the bottom-right corner belongs to the +-axis labels
            nx, ny = x1 - 40, y1 - 33
        d.text((nx, ny), "now", font=_font(11), fill=COL_PLAY)


def draw_panel(ann):
    """Render the whole right-hand panel; returns (rgb uint8 array, panel-local text records)."""
    from PIL import Image, ImageDraw

    boxes = panel_boxes()
    px0, _py0, px1, _py1 = boxes["panel"]
    img = Image.new("RGB", (PANEL_W, PANEL_H), (0, 0, 0))
    d = ImageDraw.Draw(img, "RGBA")
    d.rectangle([0, 0, PANEL_W - 1, PANEL_H - 1], fill=BG)
    panel_boxes_local = {k: (v[0] - px0, v[1], v[2] - px0, v[3]) for k, v in boxes.items()
                         if k in ("text", "full", "zoom")}
    tx0, ty0, tx1, ty1 = panel_boxes_local["text"]
    st = _Stack(d, tx0, ty0 + 1, ty1, tx1 - 2)
    draw_text_block(st, ann, panel_boxes_local)

    traces = pad_dict(ann, "pad_series")
    series = [(ch, traces[ch], PAD_COLORS[ch]) for ch in PAD_ORDER]
    nrange = nice_range(*[traces[ch] for ch in PAD_ORDER])
    frame = int(ann["frame"])
    total = episode_axis(ann)
    ticks = [(int(t["frame"]), str(t.get("text", ""))) for t in ann.get("ticks", [])]
    n = len(traces[PAD_ORDER[0]])
    for _name, arr, _c in series[1:]:
        if len(arr) != n:
            raise ValueError("the four pad traces must be the same length; %d vs %d" % (n, len(arr)))
    draw_curve(d, panel_boxes_local["full"], 0, n, total, nrange, series, ticks, frame,
               "whole episode, samples 0..%d of %d control steps" % (n - 1, total - 1))
    lo_z = max(0, frame - ZOOM_STEPS + 1)
    draw_curve(d, panel_boxes_local["zoom"], lo_z, n, ZOOM_STEPS, nrange, series, ticks, frame,
               "trailing window, steps %d..%d" % (lo_z, n - 1))
    for k in ("full", "zoom"):
        x0, y0, x1, y1 = panel_boxes_local[k]
        for j, ch in enumerate(PAD_ORDER):
            d.text((x0 + 3 + j * 46, y0 + 14), PAD_TICKER[ch], font=_font(11),
                   fill=PAD_COLORS[ch])
    return np.asarray(img), st.records


def compose_frame(front, left_wrist, ann, canvas_h=PANEL_H):
    """One composed frame, RGB uint8: [front | left_wrist | panel]. Nothing is ever drawn on a
    camera; the only thing done to a camera image is the historical 2x enlargement, in which every
    source pixel maps onto a `CAM_SCALE` x `CAM_SCALE` block of itself.
    """
    f = _enlarge(_to_uint8(front), "front")
    w = _enlarge(_to_uint8(left_wrist), "left_wrist")
    if canvas_h < CAM_H:
        raise ValueError("canvas must be at least as tall as a camera (%d), got %d"
                         % (CAM_H, canvas_h))
    panel, records = draw_panel(ann)
    canvas = np.zeros((canvas_h, 2 * CAM_W + PANEL_W, 3), dtype=np.uint8)
    cy0 = (canvas_h - CAM_H) // 2
    canvas[cy0:cy0 + CAM_H, 0:CAM_W] = f
    canvas[cy0:cy0 + CAM_H, CAM_W:2 * CAM_W] = w
    canvas[:, 2 * CAM_W:] = panel
    return canvas, [(t, x + 2 * CAM_W, y0, x1 + 2 * CAM_W, y1)
                    for (t, x, y0, x1, y1) in records]


def _enlarge(img, name):
    """Nearest-neighbour enlargement by pixel replication, not an interpolation filter.

    `cv2.resize(..., INTER_NEAREST)` gives the same picture, but this needs no cv2 and makes the
    property the selftest checks obvious: the block at rows 2y..2y+1, cols 2x..2x+1 is the source
    pixel, so the camera is recoverable verbatim by taking every other row and column.
    """
    if img.shape[:2] != (CAM_SRC, CAM_SRC):
        raise ValueError("the %s camera must be the env's native %dx%d, got %s"
                         % (name, CAM_SRC, CAM_SRC, img.shape[:2]))
    return img.repeat(CAM_SCALE, axis=0).repeat(CAM_SCALE, axis=1)


def _to_uint8(img):
    a = np.asarray(img)
    if a.dtype == np.uint8:
        return a
    a = np.nan_to_num(a.astype(np.float64))
    if a.max() <= 2.0:
        a = a * 255.0
    return np.clip(a, 0.0, 255.0).astype(np.uint8)


class PanelRecorder:
    """Streaming composer: `add(front, left_wrist, ann)` writes straight through to one mp4.

    600 control steps of 1664x576 RGB is ~1.7 GB held in Python, so nothing is accumulated; an
    episode that dies at frame 590 still leaves a watchable file.
    """

    def __init__(self, path, fps=20, canvas_h=PANEL_H):
        self.path = str(path)
        self.fps = int(fps)
        self.canvas_h = int(canvas_h)
        self.n = 0
        self.backend = None
        self._wr = None
        self._push = None
        self._close = None
        self.notes = []

    def _open(self):
        from pathlib import Path

        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        size = (2 * CAM_W + PANEL_W, self.canvas_h)
        try:
            import imageio

            wr = imageio.get_writer(self.path, fps=self.fps, codec="libx264", quality=8,
                                    macro_block_size=16, pixelformat="yuv420p")
            self.backend = "imageio/libx264/yuv420p"
            self._push = wr.append_data
            self._close = wr.close
        except Exception as exc:
            import cv2

            wr = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, size)
            if not wr.isOpened():
                raise RuntimeError("no mp4 writer available (imageio: %s; cv2 refused to open)" % exc)
            self.backend = "cv2/mp4v"
            self._push = lambda rgb: wr.write(rgb[:, :, ::-1])
            self._close = wr.release
            self.notes.append("encoded by cv2/mp4v, not the pod's libx264: layout and "
                              "synchronisation verified here, bitstream parity not")
        self._wr = wr

    def add(self, front, left_wrist, ann):
        frame = int(ann["frame"])
        if frame != self.n:
            raise RuntimeError("annotation frame %d but %d frames written; the recording is out "
                               "of order" % (frame, self.n))
        traces = pad_dict(ann, "pad_series")
        shown = pad_dict(ann, "pad_n")
        for ch in PAD_ORDER:
            s = traces[ch]
            if s.size != frame + 1:
                raise RuntimeError("pad_series[%r] holds %d samples for video frame %d -- the force "
                                   "trace must carry exactly one entry per emitted frame"
                                   % (ch, s.size, frame))
            if not np.isfinite(s).all():
                raise RuntimeError("pad_series[%r] has NaN/inf; refusing to draw a newton axis over "
                                   "a broken measurement" % ch)
            if abs(float(s[-1]) - shown[ch]) > 1e-9:
                raise RuntimeError("pad_series[%r][-1] (%.6g) is not the value displayed as this "
                                   "frame's %s (%.6g)" % (ch, float(s[-1]), ch, shown[ch]))
        img, records = compose_frame(front, left_wrist, ann, self.canvas_h)
        if self._wr is None:
            self._open()
        self._push(img)
        self.n += 1
        return img, records

    def close(self):
        import os

        if self._wr is not None:
            self._close()
        self._wr = None
        return {"path": self.path, "frames_written": self.n, "fps": self.fps,
                "backend": self.backend, "wh": [2 * CAM_W + PANEL_W, self.canvas_h],
                "bytes": os.path.getsize(self.path) if os.path.isfile(self.path) else 0,
                "notes": self.notes}


def read_frames(path):
    """Decode back to a list of RGB uint8 frames, with whichever reader is installed."""
    try:
        import imageio

        return [np.asarray(fr)[:, :, :3] for fr in imageio.get_reader(str(path))]
    except Exception:
        pass
    import cv2

    cap = cv2.VideoCapture(str(path))
    out = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        out.append(fr[:, :, ::-1].copy())
    cap.release()
    return out
