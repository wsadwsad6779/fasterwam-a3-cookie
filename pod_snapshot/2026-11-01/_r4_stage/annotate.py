import json
import numpy as np
import imageio
from PIL import Image, ImageDraw, ImageFont

BASE = "/workspace/a3_eval/r4_c0_seed22_video"
SRC = BASE + "/lora_C0_22_front.mp4"
NPZ = BASE + "/lora_C0_22.npz"
OUT = BASE + "/lora_C0_22_annotated.mp4"

m = np.load(NPZ, allow_pickle=False)
xyz = m["cookie_xyz"].astype(np.float64)
tcp = m["eef_xyz"].astype(np.float64)
f_in = m["pad_force_inner"].astype(np.float64)
f_out = m["pad_force_outer"].astype(np.float64)
changed = m["changed"].astype(bool)
opening = m["opening"].astype(np.float64)
chunk = m["chunk_idx"]
T, N, _ = xyz.shape
BX, BY, BZ, HX, HY, RIMH = 0.095, 0.100, 0.753, 0.063, 0.035, 0.026
in_bin = (np.abs(xyz[:, :, 0] - BX) < HX) & (np.abs(xyz[:, :, 1] - BY) < HY) & (xyz[:, :, 2] < BZ + RIMH)
count = in_bin.sum(1).astype(int)
print("count at 50-step marks", [int(count[t]) for t in range(49, T, 50)])
print("recomputed final", int(count[T - 1]))

entry = []
for c in range(N):
    if not in_bin[T - 1, c]:
        continue
    t = T - 1
    while t - 1 >= 0 and in_bin[t - 1, c]:
        t -= 1
    entry.append((t, c))
entry.sort()
print("entry events", entry)
by_step = dict(entry)

font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def F(sz, bold=False):
    try:
        p = font_path if not bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        return ImageFont.truetype(p, sz)
    except Exception:
        return ImageFont.load_default()


f_big, f_med, f_sml = F(20, True), F(14), F(11)
PANEL, STRIP, IMG = 300, 24, 256
W, H = IMG + PANEL + 4, IMG + STRIP + 4
DARK = (18, 20, 26)
GREEN = (60, 200, 120)
RED = (240, 90, 80)
ORANGE = (250, 170, 60)
GREY = (150, 158, 170)
WHITE = (240, 242, 246)

rd = imageio.get_reader(SRC)
wr = imageio.get_writer(OUT, fps=20, codec="libx264", pixelformat="yuv420p", quality=8, macro_block_size=16)
n = 0
for i, fr in enumerate(rd):
    t = min(i, T - 1)
    canvas = Image.new("RGB", (W, H), DARK)
    canvas.paste(Image.fromarray(np.asarray(fr).astype(np.uint8)), (2, 2))
    d = ImageDraw.Draw(canvas)
    x0 = IMG + 6
    d.text((x0, 6), "step %03d / %03d" % (t + 1, T), font=f_big, fill=WHITE)
    col = GREEN if count[t] > 0 else GREY
    d.text((x0, 32), "in box  %d / 10" % count[t], font=f_big, fill=col)
    d.text((x0, 60), "pad force  inner %6.2f N" % f_in[t], font=f_med, fill=WHITE)
    d.text((x0, 78), "           outer %6.2f N" % f_out[t], font=f_med, fill=WHITE)
    bw = PANEL - 20
    fy = 100
    d.rectangle([x0, fy, x0 + bw, fy + 14], outline=GREY)
    for k, nm in ((f_in[t], "i"), (f_out[t], "o")):
        pass
    d.rectangle([x0, fy, x0 + bw * min(1.0, float(max(f_in[t], f_out[t])) / 50.0), fy + 14],
                fill=(RED if max(f_in[t], f_out[t]) > 40 else (ORANGE if max(f_in[t], f_out[t]) > 25 else GREEN)))
    for lim, lab in ((25.0, "soft25"), (40.0, "hard40")):
        px = x0 + bw * lim / 50.0
        d.line([px, fy - 4, px, fy + 18], fill=GREY)
        d.text((px - 16, fy + 19), lab, font=f_sml, fill=GREY)
    d.text((x0, 146), "gripper opening  %.3f" % opening[t], font=f_med, fill=WHITE)
    d.text((x0, 164), "chunk %d   eef y %.3f" % (int(chunk[t]), tcp[t, 1]), font=f_med, fill=WHITE)
    st = "SAFETY REWROTE THIS STEP" if changed[t] else "model action passed through"
    d.text((x0, 190), st, font=f_med, fill=(RED if changed[t] else GREY))
    c = by_step.get(t)
    if c is not None:
        d.text((x0, 212), "cookie #%d entered the box" % c, font=f_med, fill=GREEN)
        d.rectangle([2, 2, 2 + IMG, 20], fill=(0, 0, 0))
        d.text((6, 4), "IN BOX #%d" % c, font=f_med, fill=GREEN)
    elif i - 1 in by_step or (t > 0 and (t - 1) in by_step):
        pass
    y2 = IMG + 6
    d.rectangle([2, y2, 2 + W - 4, y2 + STRIP - 6], fill=(30, 34, 44))
    for k in range(T):
        if changed[k]:
            px = 2 + (W - 4) * k / float(T)
            d.point([(px, y2 + 6), (px, y2 + 7)], fill=RED)
    for (te, cc) in entry:
        px = 2 + (W - 4) * te / float(T)
        d.line([px, y2, px, y2 + STRIP - 6], fill=GREEN)
    cur = 2 + (W - 4) * t / float(T)
    d.line([cur, y2 - 1, cur, y2 + STRIP - 5], fill=WHITE)
    wr.append_data(np.asarray(canvas))
    n += 1
rd.close()
wr.close()
import os
print("frames_written", n, "size_bytes", os.path.getsize(OUT))
