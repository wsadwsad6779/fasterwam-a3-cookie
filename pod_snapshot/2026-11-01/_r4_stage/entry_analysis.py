import json
import numpy as np

NPZ = "/workspace/a3_eval/r4_c0_seed22_video/lora_C0_22.npz"
SUM = "/workspace/a3_eval/r4_c0_seed22_video/summary_lora.json"
m = np.load(NPZ, allow_pickle=False)
xyz = m["cookie_xyz"].astype(np.float64)
xyz0 = m["cookie_xyz0"].astype(np.float64)
tcp = m["eef_xyz"].astype(np.float64)
opening = m["opening"].astype(np.float64)
app_open = m["applied_opening"].astype(np.float64)
f_in = m["pad_force_inner"].astype(np.float64)
f_out = m["pad_force_outer"].astype(np.float64)
changed = m["changed"].astype(bool)
pre = m["pre_14d"].astype(np.float64)
post = m["post_14d"].astype(np.float64)
T, N, _ = xyz.shape
BX, BY, BZ, HX, HY, RIMH = 0.095, 0.100, 0.753, 0.063, 0.035, 0.026

inside = (np.abs(xyz[:, :, 0] - BX) < HX) & (np.abs(xyz[:, :, 1] - BY) < HY)
in_bin = inside & (xyz[:, :, 2] < BZ + RIMH)
rec = [r for r in json.load(open(SUM)) if r["seed"] == 22][0]
print("T", T, "N", N, "recomputed_final_in_bin", int(in_bin[T - 1].sum()),
      "summary_in_target", rec["cookies_in_target"], "scene_ok", rec["scene_ok"])

entrants = [c for c in range(N) if in_bin[T - 1, c]]
entry = {}
for c in entrants:
    t = T - 1
    while t - 1 >= 0 and in_bin[t - 1, c]:
        t -= 1
    entry[c] = t
visitors = [c for c in range(N) if in_bin[:, c].any() and c not in entrants]
print("entrants", sorted(entrants), "entry_step", entry)
print("entered_then_left", sorted(visitors))

lift = xyz[:, :, 2] - xyz0[None, :, 2]
rel = xyz[:, :, :2] - tcp[None, :, :2]
drel = np.zeros((T, N))
drel[1:] = np.linalg.norm(np.diff(rel, axis=0), axis=2)
speed = np.zeros((T, N))
speed[1:] = np.linalg.norm(np.diff(xyz, axis=0), axis=2)
contact = (f_in > 2.0) & (f_out > 2.0)
attached = contact & (drel < 0.002)
tcp_d = np.linalg.norm(rel, axis=2)
print("max_lift_mm", round(1000 * float(lift.max()), 1))
for c in range(N):
    if lift[:, c].max() > 0.005 and not in_bin[T - 1, c]:
        pass

print("C   entry  maxlift  att_st  con_st  peak_in  peak_out  nchanged_50  d_at_entry")
for c in sorted(entrants):
    e = entry[c]
    w0 = max(0, e - 120)
    print(c, e, round(1000 * float(lift[:, c].max()), 1), int(attached[w0:e + 1, c].sum()),
          int(contact[w0:e + 1, c].sum()), round(float(f_in[w0:e + 1].max()), 2),
          round(float(f_out[w0:e + 1].max()), 2), int(changed[max(0, e - 50):e + 1].sum()),
          round(1000 * float(drel[e, c]), 3))

print()
print("per entrant trace (t, cookie xyz mm, tcp xy mm, |cookie-tcp| mm, opening, app_open, f_in, f_out, safety_rewritten)")
for c in sorted(entrants):
    e = entry[c]
    s0 = max(0, e - 100)
    first_move = int(np.argmax(speed[:, c] > 0.0005))
    print("cookie", c, "entry", e, "first_moved_step", first_move,
          "start_xy", np.round(xyz0[c, :2], 4).tolist())
    for t in range(s0, min(T, e + 6), 4):
        print("  t=%3d ck=(%6.1f,%6.1f,%6.1f) tcp=(%6.1f,%6.1f) d=%5.1f op=%6.3f aop=%6.3f fi=%6.2f fo=%6.2f ch=%d" % (
            t, 1000 * xyz[t, c, 0], 1000 * xyz[t, c, 1], 1000 * xyz[t, c, 2],
            1000 * tcp[t, 0], 1000 * tcp[t, 1], 1000 * tcp_d[t, c],
            opening[t], app_open[t], f_in[t], f_out[t], int(changed[t])))

print()
print("safety rewrite profile by channel (mean abs post-pre over all 600 steps, normalised units)")
delta = np.abs(post - pre)
print([round(float(delta[:, i].mean()), 5) for i in range(14)])
print("changed steps", int(changed.sum()), "nonzero per channel steps",
      [int((delta[:, i] > 1e-9).sum()) for i in range(14)])
print("gripper channel pre/post at changed steps: sign flip count", int(((np.sign(pre[changed][:, 6]) != np.sign(post[changed][:, 6])) & (pre[changed][:, 6] != 0)).sum()))
print("max pad force", round(max(float(f_in.max()), float(f_out.max())), 3), "hard limit 40.0")
print("steps above 25 N", int(((f_in > 25) | (f_out > 25)).sum()), "above 40 N", int(((f_in > 40) | (f_out > 40)).sum()))
print("hold windows: steps with contact", int(contact.sum()), "opening<0.6", int((opening < 0.6).sum()))
print("cookies ever lifted>5mm", int((lift.max(0) > 0.005).sum()), "of", N)
print("cookies ever moved>30mm in xy", int((np.abs(xyz[:, :, :2] - xyz0[None, :, :2]).max(2).max(1) > 0.03).sum()))
