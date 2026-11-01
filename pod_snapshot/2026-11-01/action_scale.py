import numpy as np
d = np.load("/workspace/fasterwam_a3_bridge/bridge/outputs/tf_lora/tf_lora_pm1_11.npz", allow_pickle=True)
p = d["chunk7"].astype(float)
t = d["target7"].astype(float)
v = d["valid"].astype(bool)
print("h n slope mag grip_slope grip_mag")
for h in (0, 1, 3, 7, 15, 31):
    m = v[:, h]
    pp = p[m, h, :3]
    tt = t[m, h, :3]
    slope = float((pp * tt).sum() / max((tt * tt).sum(), 1e-12))
    mag = float(np.linalg.norm(pp, axis=1).mean() / max(np.linalg.norm(tt, axis=1).mean(), 1e-12))
    gp = p[m, h, 6]
    gt = t[m, h, 6]
    gs = float((gp * gt).sum() / max((gt * gt).sum(), 1e-12))
    gm = float(np.abs(gp).mean() / max(np.abs(gt).mean(), 1e-12))
    print("%d %d %.3f %.3f %.3f %.3f" % (h, m.sum(), slope, mag, gs, gm))
pp = p[v[:, 0], 0, :3]
tt = t[v[:, 0], 0, :3]
for i in range(len(pp)):
    print("q%02d tgt_mm=%.2f pred_mm=%.2f" % (i, np.linalg.norm(tt[i]) * 1000, np.linalg.norm(pp[i]) * 1000))
