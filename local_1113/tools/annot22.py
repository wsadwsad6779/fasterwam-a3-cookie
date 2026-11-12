"""Annotated replay of a frozen r4 C0 episode: recorded front view + state.

Inputs are read-only artifacts that were already checked against the frozen run:
  * <seed>_front.mp4   -- the 600 recorded frames (sha matches the pod copy)
  * tko_mask_seed<n>.npz -- per-step mask / rules / contacts from the replay
Nothing here touches the model, wrapper, Safety or simulator.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ap = argparse.ArgumentParser()
ap.add_argument("--mp4", required=True)
ap.add_argument("--mask", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--label", required=True)
ARGS = ap.parse_args()
MP4_IN, MASK, MP4_OUT = ARGS.mp4, ARGS.mask, ARGS.out
OUT_FPS = 20
S = 2                      # front-view upscale
IMG = 256 * S
PANEL, STRIP = 342, 52
W, H = 3 + IMG + 4 + PANEL + 3, 576
DARK = (14, 16, 22)
WHITE, GREY = (236, 239, 244), (140, 148, 162)
GREEN, RED, ORANGE, YELLOW, CYAN = (58, 196, 118), (238, 88, 78), (248, 168, 58), (250, 220, 120), (86, 190, 232)
SOFT_N, HARD_N = 25.0, 40.0
BIN_XY = (0.095, 0.100)
BIN_HALF = (0.057, 0.029)
SOFT = "C:/Windows/Fonts/arial.ttf"
BOLD = "C:/Windows/Fonts/arialbd.ttf"


def F(sz, bold=False):
    return ImageFont.truetype(BOLD if bold else SOFT, sz)


f_big, f_med, f_sml = F(21, True), F(15), F(12)


def txt(dr, xy, s, font, fill):
    dr.text(xy, s, font=font, fill=fill, stroke_width=1, stroke_fill=(0, 0, 0))


d = np.load(MASK, allow_pickle=False)
mask = d["mask"].astype(bool)
traj = d["traj"].astype(float)
count = d["count"].astype(int)
eef = d["eef_xyz"].astype(float)
open_ = d["opening"].astype(float)
pad = d["pad_force"].astype(float)
contact = d["contact_any"].astype(bool)
hard = d["rule_hard"].astype(int)
soft = d["rule_soft"].astype(int)
binf = d["rule_binfloor"].astype(int)
clip = d["rule_clipped"].astype(int)
T, N, _ = traj.shape
runmax = np.maximum.accumulate(pad.max(axis=1))

entry = []
for i in np.flatnonzero(mask.any(axis=0)):
    first = int(np.argmax(mask[:, i]))
    mv = np.linalg.norm(np.diff(traj[:, i, :], axis=0), axis=1)
    fm = int(np.argmax(mv > 0.0005))
    entry.append({
        "cookie": int(i), "first_moved_step": fm, "first_in_target_step": first,
        "in_target_steps": int(mask[:, i].sum()),
        "finger_contact_steps": int(contact[:, i].sum()),
        "finger_contact_20_steps_around_entry":
            int(contact[max(0, first - 20):first + 20, i].sum()),
        "min_gap_to_cookie0_around_entry_mm":
            round(1000 * float(np.linalg.norm(
                (traj[max(0, first - 20):first + 20, i, :2]
                 - traj[max(0, first - 20):first + 20, 0, :2]), axis=1).min()), 1),
        "max_lift_mm": round(1000 * float((traj[:, i, 2] - traj[0, i, 2]).max()), 1),
        "displacement_xy_mm": round(1000 * float(np.linalg.norm(traj[-1, i, :2] - traj[0, i, :2])), 1),
        "gripper_opened_after_entry": bool(open_[first:].max() > 0.5),
    })
entry.sort(key=lambda e: e["first_in_target_step"])
by_step = {e["first_in_target_step"]: e["cookie"] for e in entry}
events = {
    "run": ARGS.label, "steps": T, "final_in_target": int(count[-1]),
    "entry_events": entry,
    "rule_totals": {"hard_estop_steps": int((hard > 0).sum()),
                    "soft_force_steps": int((soft > 0).sum()),
                    "binfloor_steps": int((binf > 0).sum()),
                    "clipped_steps": int((clip > 0).sum())},
    "bundle": {"cookies_moved_gt_10mm": [int(i) for i in range(N)
                                         if np.linalg.norm(traj[-1, i, :2] - traj[0, i, :2]) > 0.010],
               "gripper_opening_min": round(float(open_.min()), 3),
               "gripper_opening_last": round(float(open_[-1]), 3),
               "max_pad_force_N": round(float(pad.max()), 2)},
}
Path(MP4_OUT + ".events.json").write_text(json.dumps(events, indent=2))
print("EVENTS " + json.dumps(events), flush=True)

# top-down map window: source tray on the right, target bin on the left
X0, X1, Y0, Y1 = 0.015, 0.270, 0.045, 0.425
MW, MH = 322, 210


def mx(x):
    return 1 + (x - X0) / (X1 - X0) * (MW - 2)


def my(y):
    return 1 + (Y1 - y) / (Y1 - Y0) * (MH - 2)


def rule_label(t):
    if hard[t]:
        return "SAFETY HARD ESTOP  (pad force > 40 N)", RED
    if binf[t] and soft[t]:
        return "SAFETY bin-floor lift + soft", ORANGE
    if binf[t]:
        return "SAFETY bin-floor lift (hand at rim)", ORANGE
    if soft[t]:
        return "SAFETY soft force counter", ORANGE
    if clip[t]:
        return "SAFETY command clipped", CYAN
    return "model action passed through", GREY


cap = cv2.VideoCapture(MP4_IN)
fourcc = cv2.VideoWriter_fourcc(*"avc1")
wr = cv2.VideoWriter(MP4_OUT, fourcc, OUT_FPS, (W, H))
assert wr.isOpened(), "writer did not open"

# the timeline strip is static apart from the playhead, so it is drawn once
Y2 = 3 + IMG + 6
pxw = (W - 12) / float(T)
strip = Image.new("RGB", (W - 6, STRIP - 6), (26, 30, 40))
sdr = ImageDraw.Draw(strip)
for k in range(T):
    x = 6 + pxw * k
    if hard[k]:
        sdr.line([x, 2, x, 12], fill=RED)
    if soft[k]:
        sdr.line([x, 5, x, 15], fill=ORANGE)
    if binf[k]:
        sdr.line([x, 8, x, 18], fill=YELLOW)
last_lab = -99
for j, e in enumerate(entry):
    x = 6 + pxw * e["first_in_target_step"]
    sdr.line([x, 0, x, STRIP - 16], fill=GREEN)
    lx = x - 4 if x - 4 - last_lab > 13 else last_lab + 13
    sdr.text((lx, STRIP - 18), f"#{e['cookie']}", font=f_sml, fill=GREEN)
    last_lab = lx
sdr.text((6, 22), "red = hard estop    orange = soft force    yellow = bin-floor lift    "
                  "green = cookie counted inside", font=f_sml, fill=GREY)

n = 0
MAP_X = 3 + IMG + 10
MAP_Y = 292
while True:
    ok, bgr = cap.read()
    if not ok:
        break
    t = n
    if t >= T:
        break
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb).resize((IMG, IMG), Image.LANCZOS)
    canvas = Image.new("RGB", (W, H), DARK)
    canvas.paste(img, (3, 3))
    dr = ImageDraw.Draw(canvas)
    x0 = 3 + IMG + 8
    dr.text((x0, 6), ARGS.label, font=f_sml, fill=CYAN)
    dr.text((x0, 26), f"step {t + 1} / {T}", font=f_big, fill=WHITE)
    dr.text((x0, 54), f"in box  {count[t]} / 10", font=f_big,
            fill=GREEN if count[t] else GREY)
    dr.text((x0, 84), f"pad force   inner {pad[t, 0]:6.2f} N", font=f_med, fill=WHITE)
    dr.text((x0, 103), f"                  outer {pad[t, 1]:6.2f} N", font=f_med, fill=WHITE)
    bw = PANEL - 26
    fy = 126
    peak = float(max(pad[t, 0], pad[t, 1]))
    dr.rectangle([x0, fy, x0 + bw, fy + 15], outline=GREY)
    dr.rectangle([x0, fy, x0 + bw * min(1.0, peak / 50.0), fy + 15],
                 fill=RED if peak > HARD_N else (ORANGE if peak > SOFT_N else GREEN))
    for lim, lab in ((SOFT_N, "soft 25"), (HARD_N, "hard 40")):
        px = x0 + bw * lim / 50.0
        dr.line([px, fy - 4, px, fy + 19], fill=GREY)
        dr.text((px - 26, fy + 21), lab, font=f_sml, fill=GREY)
    dr.text((x0, 158), f"gripper opening {open_[t]:.3f}   eef z {eef[t, 2]:.4f} m",
            font=f_med, fill=WHITE)
    dr.text((x0, 178), f"eef x {eef[t, 0]:.3f}  y {eef[t, 1]:.3f}   peak pad {runmax[t]:.1f} N",
            font=f_sml, fill=GREY)
    rl, col = rule_label(t)
    dr.text((x0, 198), rl, font=f_med, fill=col)
    held = [int(c) for c in range(contact.shape[1]) if contact[t, c]]
    dr.text((x0, 222), f"fingers touching cookie {held if held else 'none'}",
            font=f_sml, fill=CYAN if held else GREY)
    if t in by_step:
        dr.text((x0, 242), f">> cookie #{by_step[t]} counted inside the box",
                font=f_med, fill=GREEN)
    elif t > 0 and count[t] != count[t - 1]:
        dr.text((x0, 242), f">> in-box count {count[t - 1]} -> {count[t]}",
                font=f_med, fill=GREEN if count[t] > count[t - 1] else ORANGE)

    # ---- top-down map ----
    dr.text((MAP_X, MAP_Y - 16), "top view  green=in bin  cyan=fingers  orange=moved",
            font=f_sml, fill=GREY)
    dr.rectangle([MAP_X, MAP_Y, MAP_X + MW, MAP_Y + MH], fill=(30, 34, 44), outline=GREY)
    bx0, bx1 = mx(BIN_XY[0] - BIN_HALF[0]), mx(BIN_XY[0] + BIN_HALF[0])
    by0, by1 = my(BIN_XY[1] + BIN_HALF[1]), my(BIN_XY[1] - BIN_HALF[1])
    dr.rectangle([MAP_X + bx0, MAP_Y + by0, MAP_X + bx1, MAP_Y + by1],
                 outline=GREEN if count[t] else (110, 118, 130))
    dr.text((MAP_X + bx0, MAP_Y + by1 + 1), "target bin", font=f_sml, fill=GREY)
    dr.rectangle([MAP_X + mx(0.094), MAP_Y + my(0.404), MAP_X + mx(0.256), MAP_Y + my(0.226)],
                 outline=(120, 100, 60))
    dr.text((MAP_X + mx(0.15), MAP_Y + my(0.404) + 1), "source tray", font=f_sml, fill=GREY)
    for i in range(N):
        p = traj[t, i]
        cx, cy = MAP_X + mx(p[0]), MAP_Y + my(p[1])
        if mask[t, i]:
            dr.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], fill=GREEN)
        elif contact[t, i]:
            dr.ellipse([cx - 3, cy - 3, cx + 3, cy + 3], fill=CYAN)
        elif np.linalg.norm(p[:2] - traj[0, i, :2]) > 0.010:
            dr.ellipse([cx - 3, cy - 3, cx + 3, cy + 3], outline=ORANGE)
        else:
            dr.ellipse([cx - 1, cy - 1, cx + 1, cy + 1], fill=(96, 84, 52))
    ex, ey = MAP_X + mx(eef[t, 0]), MAP_Y + my(eef[t, 1])
    dr.line([ex - 6, ey, ex + 6, ey], fill=RED)
    dr.line([ex, ey - 6, ex, ey + 6], fill=RED)

    # ---- in-image badge, top-right (over the background, never the arm) ----
    badge = f"in box {count[t]}/10   step {t + 1}"
    bw2 = int(dr.textlength(badge, font=f_sml))
    bx = 3 + IMG - 8 - bw2
    txt(dr, (bx, 10), badge, f_sml, GREEN if count[t] else WHITE)

    # ---- timeline strip (static base + playhead) ----
    canvas.paste(strip, (3, Y2))
    dr = ImageDraw.Draw(canvas)
    cur = 9 + pxw * t
    dr.line([cur, Y2 - 1, cur, Y2 + STRIP - 9], fill=WHITE)

    wr.write(cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR))
    n += 1
    if n % 100 == 0:
        print("frame", n, flush=True)
cap.release()
wr.release()
print("frames_written", n, "size_bytes", Path(MP4_OUT).stat().st_size, flush=True)
