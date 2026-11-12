"""Verdict on the cookies that scored inside the bin in a B0 (bare) episode.

Reads only the read-only replay npz produced by tko_mask.py, so nothing here
re-runs the simulator. The question it answers: is a counted cookie actually
resting in the bin, or is it still held by the gripper / crushed against the
wall, and what force was on it when it got there.

    python3 mech_b0.py --mask tko_mask_B0_22.npz --bin-t 60
"""
import argparse

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--mask", required=True)
ap.add_argument("--bin-t", type=int, default=50, help="timeline print every N steps")
a = ap.parse_args()

d = np.load(a.mask, allow_pickle=False)
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
T, N, _ = traj.shape
BIN = np.array([0.095, 0.100])
HALF = np.array([0.057, 0.029])
RIM_Z = 0.753 + 0.026
FLOOR = 0.775

print("T=%d N=%d final_count=%d max_count=%d" % (T, N, count[-1], count.max()))
inside = [int(i) for i in range(N) if mask[:, i].any()]
held_final = [int(i) for i in range(N) if mask[-1, i]]
print("ever counted inside:", inside)
print("counted inside at the LAST step:", held_final)

pk = int(pad.argmax())
print("peak pad %.2f N at t=%d (eef=%s d_bin=%.1fmm open=%.3f count=%d)"
      % (pad[pk, 0] if pad[pk, 0] >= pad[pk, 1] else pad[pk, 1], pk,
         np.round(eef[pk], 4).tolist(), 1000 * np.linalg.norm(eef[pk, :2] - BIN),
         open_[pk], count[pk]))
for thr in (25, 40, 100, 300):
    print("  pad>%3dN steps=%-4d first=%s" % (thr, int((pad.max(axis=1) > thr).sum()),
                                              np.flatnonzero(pad.max(axis=1) > thr)[:5].tolist()))

for i in inside:
    z = traj[:, i, 2]
    rel = traj[:, i, :2] - eef[:, :2]
    de = np.linalg.norm(rel, axis=1)
    rigid = int((np.linalg.norm(np.diff(rel, axis=0), axis=1) < 0.002).sum())
    dxy_final = np.abs(traj[-1, i, :2] - BIN)
    mv = np.linalg.norm(np.diff(traj[:, i, :], axis=0), axis=1)
    fm = int(np.argmax(mv > 0.0005)) if (mv > 0.0005).any() else -1
    ent = int(np.argmax(mask[:, i])) if mask[:, i].any() else -1
    ver = np.linalg.norm(traj[-1, i] - traj[-2, i])
    print("\n-- cookie %d entry=%s first_move=%s contact=%d/%d" % (i, ent, fm, int(contact[:, i].sum()), T))
    print("   final pos=%s in_footprint=%s (|dx|=%.1fmm |dy|=%.1fmm of %.0f/%.0f)"
          % (np.round(traj[-1, i], 4).tolist(),
             bool(np.all(dxy_final < HALF)),
             1000 * dxy_final[0], 1000 * dxy_final[1], 1000 * HALF[0], 1000 * HALF[1]))
    print("   final z=%.4f (floor_top %.3f rim %.3f)  lift_max=%.1fmm  xy_disp=%.1fmm"
          % (z[-1], FLOOR, RIM_Z, 1000 * (z.max() - z[0]),
             1000 * np.linalg.norm(traj[-1, i, :2] - traj[0, i, :2])))
    print("   last-step displacement=%.2fmm  last-step |v|=%.4f m/s (settle needs <0.06)"
          % (1000 * ver, ver))
    print("   min dist to TCP=%.1fmm  within20mm=%d  rigid_with_TCP=%d  final_open=%.3f  open@[entry]=%s"
          % (1000 * de.min(), int((de < 0.020).sum()), rigid, open_[-1],
             None if ent < 0 else round(float(open_[ent]), 3)))
    if ent >= 0:
        w = slice(max(0, ent - 30), min(T, ent + 3))
        print("   30 steps before entry: pad_max=%.1fN hard=%d soft=%d binfloor=%d eef_z_min=%.4f open[min=%.3f max=%.3f]"
              % (pad[w].max(), int(hard[w].sum()), int(soft[w].sum()), int(binf[w].sum()),
                 eef[w, 2].min(), open_[w].min(), open_[w].max()))

print("\n== count timeline (every %d steps)" % a.bin_t)
for t in range(0, T, a.bin_t):
    print("t=%3d count=%d inside=%s eef=%s d_bin=%5.1fmm pad=(%.1f,%.1f) open=%.3f hard=%d soft=%d binf=%d"
          % (t, count[t], [int(c) for c in np.flatnonzero(mask[t])],
             np.round(eef[t], 3).tolist(), 1000 * np.linalg.norm(eef[t, :2] - BIN),
             pad[t, 0], pad[t, 1], open_[t], hard[t], soft[t], binf[t]))

print("\n== last 25 steps")
for t in range(max(0, T - 25), T):
    print("t=%3d count=%d inside=%s eef_z=%.4f pad=(%.1f,%.1f) open=%.3f"
          % (t, count[t], [int(c) for c in np.flatnonzero(mask[t])], eef[t, 2],
             pad[t, 0], pad[t, 1], open_[t]))
