import numpy as np
r = np.load("/workspace/fasterwam_a3_bridge/bridge/outputs/C4/lora_C4_11.npz", allow_pickle=True)
e = np.load("/workspace/expert_seed11.npz", allow_pickle=True)
eef = r["eef_xyz"].astype(float)
d = np.zeros_like(eef)
d[1:] = np.linalg.norm(np.diff(eef, axis=0), axis=1)[:, None]
post = r["post_14d"].astype(float)
postn = np.linalg.norm(post[:, :3], axis=1)
ph = r["phase"].astype(int)
ci = r["chunk_idx"].astype(int)
print("chunk_idx unique", np.unique(ci)[:12], "counts", np.bincount(ci)[:12])
for p in np.unique(ph):
    m = ph == p
    print("run phase %d n=%d eef_step_mm=%.2f model_act_mm=%.2f" % (p, m.sum(), d[m, 0].mean() * 1000, postn[m].mean() * 1000))
ea = np.asarray(e["act14"], dtype=float)
ex = np.asarray(e["eef"], dtype=float)[:, :3]
ed = np.linalg.norm(np.diff(ex, axis=0), axis=1) * 1000
eph = [x.decode() if isinstance(x, bytes) else str(x) for x in np.asarray(e["phases"]).reshape(-1)]
for name in ["SELECT_COOKIE", "DESCEND", "CLOSE", "LIFT", "MOVE_TO_SLOT"]:
    m = np.array([q == name for q in eph])[:-1]
    print("exp %s n=%d eef_step_mm=%.2f act_mm=%.2f" % (name, m.sum(), ed[m].mean(), np.linalg.norm(ea[:len(m)][m][:, :3], axis=1).mean() * 1000))
