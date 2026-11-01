import numpy as np, glob, os
for p in sorted(glob.glob("/workspace/fasterwam_a3_bridge/bridge/outputs/*/lora_*_1*.npz")):
    tag = p.split("outputs/")[1][:-4]
    seed = int(tag.split("_")[-1])
    epf = "/workspace/expert_seed%d.npz" % seed
    if not os.path.exists(epf): print("SKIP", tag); continue
    R = np.load(p, allow_pickle=True); E = np.load(epf, allow_pickle=True)
    ep = E["eef"][:, :3].astype(np.float64); rp = R["eef_xyz"].astype(np.float64)
    a = ep[:-1]; ab = ep[1:] - a; L2 = (ab*ab).sum(1)
    P = rp[:, None, :]
    t = np.clip(((P - a[None])*ab[None]).sum(-1)/np.where(L2 == 0, 1.0, L2)[None], 0.0, 1.0)
    q = a[None] + t[..., None]*ab[None]
    dist = np.linalg.norm(P - q, axis=-1)
    j = dist.argmin(1); res = dist.min(1); cnt = j + t[np.arange(len(rp)), j]
    tail = rp[-200:]
    pad = np.maximum(np.abs(R["pad_force_inner"]), np.abs(R["pad_force_outer"]))
    print(tag, "prog%%", round(float(cnt[-1])/(len(ep)-1)*100, 1), "offpath_mm", round(float(np.median(res))*1000, 1), "tailspan_mm", round(float(np.linalg.norm(tail.max(0)-tail.min(0)))*1000, 1), "padmax", round(float(pad.max()), 1), "pad>10N%%", round(100*float((pad>10).mean()), 1), "open", round(float(R["opening"].mean()), 3), "dsrc_end", round(float(R["dist_to_source"][-1]), 3))
