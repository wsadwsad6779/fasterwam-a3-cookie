"""Compose an annotated replay: front-camera frames + authoritative per-step state.

Reads only: the recorded mp4 and the mask npz written by tko_mask.py (whose
cookie trajectory and Safety output were checked against the run npz). Nothing
here can change what was executed.

    python3 tko_annot.py --mp4 <front.mp4> --mask <mask.npz> --out <annotated.mp4>
                         --label lora_C0_22 --scale 2
"""
import argparse
import json
from pathlib import Path

import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_DIR = "/usr/share/fonts/truetype/dejavu"
SOFT_N = 25.0
HARD_N = 40.0


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mp4", required=True)
    ap.add_argument("--mask", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--scale", type=int, default=2)
    return ap.parse_args()


def main():
    args = parse_args()
    d = np.load(args.mask, allow_pickle=False)
    count = d["count"].astype(int)
    hard = d["rule_hard"].astype(int)
    soft = d["rule_soft"].astype(int)
    binfloor = d["rule_binfloor"].astype(int)
    clipped = d["rule_clipped"].astype(int)
    contact = d["contact_any"].astype(bool)
    pad = d["pad_force"].astype(float)
    eef = d["eef_xyz"].astype(float)
    opening = d["opening"].astype(float)
    mask = d["mask"].astype(bool)
    traj = d["traj"].astype(float)
    T = len(count)

    entry = []
    for i in range(mask.shape[1]):
        if not mask[:, i].any():
            continue
        first = int(np.argmax(mask[:, i]))
        moved = np.linalg.norm(np.diff(traj[:, i, :], axis=0), axis=1)
        first_move = int(np.argmax(moved > 0.0005)) if (moved > 0.0005).any() else -1
        entry.append({
            "cookie": i,
            "first_moved_step": first_move,
            "first_in_target_step": first,
            "in_target_steps": int(mask[:, i].sum()),
            "finger_contact_steps": int(contact[:, i].sum()),
            "contact_at_entry": int(contact[max(0, first - 10):first + 1, i].sum()),
            "max_lift_mm": round(1000 * float((traj[:, i, 2] - traj[0, i, 2]).max()), 1),
            "displacement_xy_mm": round(1000 * float(np.linalg.norm(
                traj[T - 1, i, :2] - traj[0, i, :2])), 1),
            "peak_pad_force_near_move": round(float(
                pad[max(0, first_move - 5):first + 1].max()), 2) if first_move >= 0 else None,
        })
    entry.sort(key=lambda e: e["first_in_target_step"])
    by_step = {e["first_in_target_step"]: e["cookie"] for e in entry}
    events = {"label": args.label, "steps": T, "final_in_target": int(count[T - 1]),
              "entry_events": entry,
              "rule_totals": {"hard": int(hard.sum()), "soft": int(soft.sum()),
                              "binfloor": int(binfloor.sum()), "clipped": int(clipped.sum())},
              "max_pad_force": round(float(pad.max()), 3)}
    Path(args.out + ".events.json").write_text(json.dumps(events, indent=2))
    print("EVENTS " + json.dumps(events), flush=True)

    def F(sz, bold=False):
        name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
        return ImageFont.truetype(f"{FONT_DIR}/{name}", sz)

    f_big, f_med, f_sml = F(22, True), F(15), F(12)
    S = args.scale
    IMG = 256 * S
    PANEL, STRIP = 320, 40
    W, H = IMG + PANEL + 6, IMG + STRIP + 6
    DARK, WHITE, GREY = (16, 18, 24), (238, 241, 246), (146, 154, 168)
    GREEN, RED, ORANGE, CYAN = (58, 196, 118), (238, 88, 78), (248, 168, 58), (86, 190, 232)

    def rule_label(t):
        if hard[t]:
            return "SAFETY: HARD ESTOP (force > 40 N)", RED
        if binfloor[t] and soft[t]:
            return "SAFETY: bin-floor + soft retreat", ORANGE
        if binfloor[t]:
            return "SAFETY: bin-floor lift (hand in box)", ORANGE
        if soft[t]:
            return "SAFETY: soft force counter", ORANGE
        if clipped[t]:
            return "SAFETY: command clipped", CYAN
        return "model action passed through", GREY

    rd = imageio.get_reader(args.mp4)
    wr = imageio.get_writer(args.out, fps=20, codec="libx264", pixelformat="yuv420p",
                            quality=8, macro_block_size=16)
    n = 0
    for i, fr in enumerate(rd):
        if i >= T:
            break
        t = i
        img = Image.fromarray(np.asarray(fr).astype(np.uint8)).resize((IMG, IMG), Image.LANCZOS)
        canvas = Image.new("RGB", (W, H), DARK)
        canvas.paste(img, (3, 3))
        dr = ImageDraw.Draw(canvas)
        x0 = IMG + 8
        dr.text((x0, 8), args.label, font=f_med, fill=CYAN)
        dr.text((x0, 28), f"step {t + 1} / {T}", font=f_big, fill=WHITE)
        dr.text((x0, 56), f"in box  {count[t]} / 10", font=f_big,
                fill=GREEN if count[t] else GREY)
        dr.text((x0, 86), f"pad force  inner {pad[t, 0]:6.2f} N", font=f_med, fill=WHITE)
        dr.text((x0, 105), f"           outer {pad[t, 1]:6.2f} N", font=f_med, fill=WHITE)
        bw = PANEL - 24
        fy = 128
        peak = float(max(pad[t, 0], pad[t, 1]))
        dr.rectangle([x0, fy, x0 + bw, fy + 16], outline=GREY)
        dr.rectangle([x0, fy, x0 + bw * min(1.0, peak / 50.0), fy + 16],
                     fill=RED if peak > HARD_N else (ORANGE if peak > SOFT_N else GREEN))
        for lim, lab in ((SOFT_N, "soft 25"), (HARD_N, "hard 40")):
            px = x0 + bw * lim / 50.0
            dr.line([px, fy - 5, px, fy + 21], fill=GREY)
            dr.text((px - 22, fy + 23), lab, font=f_sml, fill=GREY)
        dr.text((x0, 162), f"gripper opening {opening[t]:.3f}", font=f_med, fill=WHITE)
        dr.text((x0, 181), f"eef  x {eef[t, 0]:.3f}  y {eef[t, 1]:.3f}  z {eef[t, 2]:.3f}",
                font=f_sml, fill=WHITE)
        held = [int(c) for c in range(contact.shape[1]) if contact[t, c]]
        rl, col = rule_label(t)
        dr.text((x0, 202), rl, font=f_med, fill=col)
        dr.text((x0, 226), f"finger contact: {held if held else 'none'}", font=f_sml,
                fill=CYAN if held else GREY)
        if t in by_step:
            dr.text((x0, 248), f"cookie #{by_step[t]} entered the box", font=f_med, fill=GREEN)
        if t > 0 and count[t] > count[t - 1]:
            dr.text((x0, 270), f"+{count[t] - count[t - 1]} this step", font=f_med, fill=GREEN)
        dr.rectangle([3, 3, 3 + 130, 24], fill=(0, 0, 0))
        dr.text((8, 5), f"in box {count[t]}/10", font=f_sml, fill=GREEN if count[t] else GREY)
        y2 = IMG + 8
        dr.rectangle([3, y2, 3 + W - 6, y2 + STRIP - 4], fill=(28, 32, 42))
        pxw = (W - 6) / float(T)
        for k in range(T):
            x = 3 + pxw * k
            if hard[k]:
                dr.line([x, y2 + 2, x, y2 + 12], fill=RED)
            if soft[k]:
                dr.line([x, y2 + 4, x, y2 + 14], fill=ORANGE)
            if binfloor[k]:
                dr.line([x, y2 + 6, x, y2 + 16], fill=(250, 220, 120))
        for e in entry:
            x = 3 + pxw * e["first_in_target_step"]
            dr.line([x, y2, x, y2 + STRIP - 6], fill=GREEN)
            dr.text((x + 1, y2 + STRIP - 18), f"#{e['cookie']}", font=f_sml, fill=GREEN)
        cur = 3 + pxw * t
        dr.line([cur, y2 - 1, cur, y2 + STRIP - 5], fill=WHITE)
        dr.text((6, y2 + 18), "red=hard estop  orange=soft  yellow=bin-floor  green=cookie entered",
                font=f_sml, fill=GREY)
        wr.append_data(np.asarray(canvas))
        n += 1
    rd.close()
    wr.close()
    print("frames_written", n, "size_bytes", Path(args.out).stat().st_size, flush=True)


if __name__ == "__main__":
    main()
