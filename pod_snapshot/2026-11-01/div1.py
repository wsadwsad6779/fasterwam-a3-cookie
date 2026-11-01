import numpy as np
E = np.load("/workspace/expert_seed11.npz", allow_pickle=True)
R = np.load("/workspace/fasterwam_a3_bridge/bridge/outputs/C5/lora_C5_11.npz", allow_pickle=True)
print("R_KEYS", sorted(R.files))
ep = E["eef"][:, :3]; eo = E["opening"]; eph = E["phases"]
rp = R["eef_xyz"]; ro = R["opening"]; rph = R["phase"]
pad = np.maximum(np.abs(R["pad_force_inner"]), np.abs(R["pad_force_outer"]))
n = min(len(ep), len(rp), len(ro))
print("N", len(ep), len(rp))
d = np.linalg.norm(rp[:n] - ep[:n], axis=1)
print("pos_err_mm step: 0,1,2,5,10,20,40,80,120,200,300,400,500", [round(float(d[i])*1000,1) for i in [0,1,2,5,10,20,40,80,120,200,300,400,min(500,n-1)]])
first = next((i for i in range(n) if d[i] > 0.005), -1)
print("FIRST_OVER_5MM", first)
for i in [0, first, first+10, first+50, first+150]:
    if i < 0 or i >= n: continue
    print(" i=%d rp=%s ep=%s |d|=%.1fmm open r=%.3f e=%.3f pad=%.1fN ph r=%s e=%s" % (i, np.round(rp[i],3), np.round(ep[i],3), d[i]*1000, ro[i], eo[i], pad[i], rph[i], eph[i]))
de = np.linalg.norm(np.diff(rp, axis=0), axis=1); ee = np.linalg.norm(np.diff(ep, axis=0), axis=1)
print("step_len_mm rollout med/max", round(float(np.median(de))*1000,2), round(float(de.max())*1000,2))
print("step_len_mm expert  med/max", round(float(np.median(ee))*1000,2), round(float(ee.max())*1000,2))
k = min(len(de), len(ee))
cos = (de[:k]*ee[:k] > 0).sum()/k
print("same_sign_motion_pct", round(100*float(cos),1))
print("pad_force max/med", round(float(pad.max()),1), round(float(np.median(pad)),1))
print("open_abs_mean r/e", round(float(np.abs(ro).mean()),3), round(float(np.abs(eo).mean()),3))
