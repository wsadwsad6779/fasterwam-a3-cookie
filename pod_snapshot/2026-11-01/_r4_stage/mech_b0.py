"""Verdict on the cookies that scored inside the bin in a B0 (bare) episode."""
import argparse

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--mask", required=True)
ap.add_argument("--bin-t", type=int, default=60)
a = ap.parse_args()

d = np.load(a.mask, allow_pickle=False)
mask = d["mask"].astype(bool)
traj = d["traj"].astype(float)
count = d["count"].astype(int)
eef = d["eef_xyz"].astype(float)
open_ = d["opening"].astype(float)
pad = d["pad_force"].astype(float)
contact = d["contact_any"].astype(bool)
T, N, _ = traj.shape
BIN = np.array([0.095, 0.100])
HALF = np.array([0.057, 0.029])
RIM_Z = 0.753 + 0.026
FLOOR = 0.775

print("T=%d N=%d final_count=%d max_count=%d" % (T, N, count[-1], count.max()))
inside = [int(i) for i in range(N) if mask[:, i].any()]
print("ever counted inside:", inside)
print("counted at LAST step:", [int(i) for i in range(N) if mask[-1, i]])
pk = int(pad.max(axis=1).argmax())
print("peak pad %.2fN at t=%d eef=%s d_bin=%.1fmm open=%.3f count=%d"
      % (pad[pk].max(), pk, np.round(eef[pk], 4).tolist(),
         1000 * np.linalg.norm(eef[pk, :2] - BIN), open_[pk], count[pk]))
for thr in (25, 40, 100, 300, 600):
    idx = np.flatnonzero(pad.max(axis=1) > thr)
    print("  pad>%3dN steps=%-4d first=%s last=%s" % (thr, len(idx), idx[:5].tolist(), idx[-3:].tolist()))
for i in inside:
    z = traj[:, i, 2]
    rel = traj[:, i, :2] - eef[:, :2]
    de = np.linalg.norm(rel, axis=1)
    rigid = int((np.linalg.norm(np.diff(rel, axis=0), axis=1) < 0.002).sum())
    dxy = np.abs(traj[-1, i, :2] - BIN)
    mv = np.linalg.norm(np.diff(traj[:, i, :], axis=0), axis=1)
    fm = int(np.argmax(mv > 0.0005)) if (mv > 0.0005).any() else -1
    ent = int(np.argmax(mask[:, i])) if mask[:, i].any() else -1
    ver = float(np.linalg.norm(traj[-1, i] - traj[-2, i]))
    print()
    print("-- cookie %d entry=%s first_move=%s finger_contact=%d/%d" % (i, ent, fm, int(contact[:, i].sum()), T))
    print("   final=%s in_footprint=%s (|dx|=%.1f/%.0fmm |dy|=%.1f/%.0fmm)"
          % (np.round(traj[-1, i], 4).tolist(), bool(np.all(dxy < HALF)),
             1000 * dxy[0], 1000 * HALF[0], 1000 * dxy[1], 1000 * HALF[1]))
    print("   final_z=%.4f (floor %.3f rim %.3f) lift=%.1fmm xy_disp=%.1fmm last_disp=%.2fmm"
          % (z[-1], FLOOR, RIM_Z, 1000 * (z.max() - z[0]),
             1000 * np.linalg.norm(traj[-1, i, :2] - traj[0, i, :2]), 1000 * ver))
    print("   min_TCP=%.1fmm within20mm=%d rigid_with_TCP=%d final_open=%.3f open_at_entry=%s"
          % (1000 * de.min(), int((de < 0.020).sum()), rigid, open_[-1],
             None if ent < 0 else round(float(open_[ent]), 3)))
    if ent >= 0:
        w = slice(max(0, ent - 30), min(T, ent + 3))
        print("   30 steps before entry: pad_max=%.1fN eef_z_min=%.4f open[min=%.3f max=%.3f]"
              % (pad[w].max(), eef[w, 2].min(), open_[w].min(), open_[w].max()))
print()
print("== count timeline every %d steps" % a.bin_t)
for t in range(0, T, a.bin_t):
    print("t=%3d count=%d inside=%s eef=%s d_bin=%5.1fmm pad=(%.1f,%.1f) open=%.3f"
          % (t, count[t], [int(c) for c in np.flatnonzero(mask[t])],
             np.round(eef[t], 3).tolist(), 1000 * np.linalg.norm(eef[t, :2] - BIN),
             pad[t, 0], pad[t, 1], open_[t]))
print()
print("== last 20 steps")
for t in range(max(0, T - 20), T):
    print("t=%3d count=%d inside=%s eef_z=%.4f pad=(%.1f,%.1f) open=%.3f"
          % (t, count[t], [int(c) for c in np.flatnonzero(mask[t])], eef[t, 2],
             pad[t, 0], pad[t, 1], open_[t]))
