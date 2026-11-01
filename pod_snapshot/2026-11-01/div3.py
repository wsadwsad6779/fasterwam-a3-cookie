import numpy as np
E = np.load("/workspace/expert_seed11.npz", allow_pickle=True)
R = np.load("/workspace/fasterwam_a3_bridge/bridge/outputs/C5/lora_C5_11.npz", allow_pickle=True)
ep = E["eef"][:, :3]; rp = R["eef_xyz"]
d = np.linalg.norm(np.diff(ep, axis=0), axis=1); s = np.concatenate([[0.0], np.cumsum(d)])
def proj(p):
    best = None
    for j in range(len(ep)-1):
        a, b = ep[j], ep[j+1]; ab = b - a; L2 = float(ab @ ab)
        t = 0.0 if L2 == 0 else float(np.clip((p - a) @ ab / L2, 0.0, 1.0))
        q = a + t*ab; dd = float(np.linalg.norm(p - q))
        if best is None or dd < best[0]: best = (dd, j + t)
    return best
idx = [proj(p) for p in rp]
cnt = np.array([i[1] for i in idx]); resid = np.array([i[0] for i in idx])
print("expert_path_len_mm", round(float(s[-1])*1000,1), "steps", len(ep))
print("prog_frac_at", [(k, round(float(cnt[k])/(len(ep)-1),3)) for k in [0,50,100,200,300,400,500,599]])
print("offpath_resid_mm_at", [(k, round(float(resid[k])*1000,1)) for k in [0,50,100,200,300,400,500,599]])
print("offpath_resid_mm median/p90/max", round(float(np.median(resid))*1000,1), round(float(np.percentile(resid,90))*1000,1), round(float(resid.max())*1000,1))
tail = rp[-200:]
print("last200_xyz_std_mm", np.round(tail.std(axis=0)*1000,1), "span_mm", np.round((tail.max(axis=0)-tail.min(axis=0))*1000,1))
ds = R["dist_to_source"]
print("dist_to_source start/end/min/max", round(float(ds[0]),3), round(float(ds[-1]),3), round(float(ds.min()),3), round(float(ds.max()),3))
pi = np.abs(R["pad_force_inner"]); po = np.abs(R["pad_force_outer"]); pad = np.maximum(pi, po)
print("pad_over_30N_pct", round(100*float((pad>30).mean()),1), "over_10N_pct", round(100*float((pad>10).mean()),1))
ro = R["opening"]
print("open_min/mean/max", round(float(ro.min()),3), round(float(ro.mean()),3), round(float(ro.max()),3))
