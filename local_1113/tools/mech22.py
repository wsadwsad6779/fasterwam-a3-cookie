"""Mechanism analysis of the three cookies that scored inside the bin (seed 22).

Reads only the read-only replay npz (tko_mask_seed22.npz), whose cookie
trajectory and Safety output were already shown to match the frozen run npz
bit-for-bit. Nothing here re-runs the simulator.
"""
import json

import numpy as np

d = np.load("tko_mask_seed22.npz", allow_pickle=False)
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
BIN = np.array([0.095, 0.100])
RIM_Z = 0.753 + 0.026

print("T", T, "N", N, "final count", count[-1], "max count", count.max())
p0 = traj[0]
print("source xy range x[%.3f,%.3f] y[%.3f,%.3f] z %s"
      % (p0[:, 0].min(), p0[:, 0].max(), p0[:, 1].min(), p0[:, 1].max(),
         sorted(set(np.round(p0[:, 2], 4).tolist()))[:4]))
uniq = np.unique(np.round(p0[:, :2], 4), axis=0)
print("unique source xy slots:", len(uniq))
dxy = np.linalg.norm(np.diff(uniq, axis=0), axis=1) if len(uniq) > 1 else []
print("slot spacing mm:", np.round(1000 * dxy, 1)[:12])

print("\n== eef path (every 50 steps): xy dist to bin centre, z, opening, pad max")
for t in range(0, T, 50):
    print("t=%3d d_bin=%5.1fmm z=%.4f open=%.3f pad=(%.2f,%.2f) in_bin=%d"
          % (t, 1000 * np.linalg.norm(eef[t, :2] - BIN), eef[t, 2], open_[t],
             pad[t, 0], pad[t, 1], int(mask[t].sum())))

print("\n== rules: steps where each counter incremented")
for name, arr in (("hard", hard), ("soft", soft), ("binfloor", binf), ("clipped", clip)):
    idx = np.flatnonzero(arr)
    print("%-9s n=%-4d first=%s last=%s" % (name, len(idx), idx[:6].tolist(),
                                            idx[-4:].tolist() if len(idx) else []))

inside = [int(i) for i in range(N) if mask[:, i].any()]
movers = [int(i) for i in range(N)
          if np.abs(traj[:, i, 2] - traj[0, i, 2]).max() > 0.003
          or np.linalg.norm(traj[-1, i, :2] - traj[0, i, :2]) > 0.01]
print("\ncookies ever counted inside:", inside)
print("cookies that moved:", movers)

for i in movers:
    mv = np.linalg.norm(np.diff(traj[:, i, :], axis=0), axis=1)
    fm = int(np.argmax(mv > 0.0005)) if (mv > 0.0005).any() else -1
    ent = int(np.argmax(mask[:, i])) if mask[:, i].any() else -1
    z = traj[:, i, 2]
    # gripper-relative behaviour
    rel = traj[:, i, :2] - eef[:, :2]
    de = np.linalg.norm(rel, axis=1)
    rigid_e = np.linalg.norm(np.diff(rel, axis=0), axis=1) < 0.002
    # nearest other cookie (excluding self) at each step
    pts = traj[:, :, :2]
    dd = np.linalg.norm(pts - pts[:, i:i + 1, :], axis=2)
    dd[:, i] = 9e9
    near = dd.argmin(axis=1)
    near_d = dd.min(axis=1)
    win = slice(max(0, ent - 30), min(T, ent + 3)) if ent >= 0 else slice(0, 0)
    print("\n-- cookie %d  first_move=%d entry=%s contact=%d/%d" %
          (i, fm, ent, int(contact[:, i].sum()), T))
    print("   lift max=%.1fmm final_z=%.4f (rim %.3f) xy_disp=%.1fmm final_xy=%s"
          % (1000 * (z.max() - z[0]), z[-1], RIM_Z,
             1000 * np.linalg.norm(traj[-1, i, :2] - traj[0, i, :2]),
             np.round(traj[-1, i, :2], 4).tolist()))
    print("   min dist to TCP=%.1fmm  steps within 20mm of TCP=%d  rigid-with-TCP steps=%d"
          % (1000 * de.min(), int((de < 0.020).sum()), int(rigid_e.sum())))
    if ent >= 0:
        print("   in the 30 steps before entry: binfloor=%d hard=%d pad_max=%.2fN "
              "open[min=%.3f,max=%.3f] eef_z_min=%.4f"
              % (int(binf[win].sum()), int(hard[win].sum()), pad[win].max(),
                 open_[win].min(), open_[win].max(), eef[win, 2].min()))
    print("   nearest cookie timeline (step: which, gap mm):",
          [(int(t), int(near[t]), round(1000 * float(near_d[t]), 1))
           for t in ([fm, fm + 60, 300, 420, 460, 480, 500, 560, T - 1]
                     if fm >= 0 else [300, 460, 480, 500, T - 1])])
    print("   ever gripped (finger contact) steps:", np.flatnonzero(contact[:, i])[:5].tolist(),
          "...", np.flatnonzero(contact[:, i])[-3:].tolist())
    d0 = np.linalg.norm(traj[:, :, :2] - traj[:, 0:1, :2], axis=2)[:, i]
    d0z = np.abs(traj[:, :, 2] - traj[:, 0:1, 2])[:, i]
    print("   initial slot distance to cookie 0 = %.1f mm"
          % (1000 * float(np.linalg.norm(traj[0, i, :2] - traj[0, 0, :2]))))
    if ent >= 0:
        w2 = slice(max(0, ent - 20), min(T, ent + 20))
        print("   around entry +-20: gap to cookie0 min=%.1fmm dz min=%.1fmm "
              "finger contact steps=%d TCP gap min=%.1fmm"
              % (1000 * d0[w2].min(), 1000 * d0z[w2].min(),
                 int(contact[w2, i].sum()), 1000 * de[w2].min()))
        print("   transport phase (t=360..470): mean gap to cookie0=%.1fmm moved_with_0=%d/110"
              % (1000 * float(d0[360:470].mean()),
                 int((np.linalg.norm(np.diff(traj[360:470, i, :2], axis=0)
                                     - np.diff(traj[360:470, 0, :2], axis=0), axis=1) < 0.002).sum())))

print("\n== count timeline around the entries")
for t in range(470, 505):
    print("t=%3d count=%d inside=%s eef_z=%.4f binfloor=%d hard=%d pad=(%.1f,%.1f) open=%.3f"
          % (t, count[t], [int(c) for c in np.flatnonzero(mask[t])], eef[t, 2],
             binf[t], hard[t], pad[t, 0], pad[t, 1], open_[t]))
