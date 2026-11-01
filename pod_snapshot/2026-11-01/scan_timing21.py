import numpy as np
d = np.load("/workspace/fasterwam_a3_bridge/bridge/outputs/r3/C0/lora_C0_21.npz")
pre = np.asarray(d["pre_14d"], dtype=np.float64)
post = np.asarray(d["post_14d"], dtype=np.float64)
ch = np.asarray(d["changed"])
ci = np.asarray(d["chunk_idx"])
op = np.asarray(d["opening"], dtype=np.float64)
eef = np.asarray(d["eef_xyz"], dtype=np.float64)
T = len(pre)
print("T", T, "safety_changed", int(ch.sum()), "pct", round(100*ch.mean(),2))
# chunk boundaries
bnd = [i for i in range(1,T) if ci[i] != ci[i-1]]
print("chunks", len(set(ci)), "boundaries_first20", bnd[:20])
# find sign flips of pre gripper
sg = np.sign(pre[:,6])
flips = [i for i in range(1,T) if sg[i] != sg[i-1]]
print("gripper_flips", flips[:20])
open_steps = [i for i in range(1,T) if pre[i,6] > 0.5 and pre[i-1,6] <= 0.5]
print("open_events", open_steps)
for t in range(170, 262):
    print(t, "ci", int(ci[t]), "pre_g", round(float(pre[t,6]),2), "post_g", round(float(post[t,6]),2), "cmd", round(float(d["applied_canonical"][t,7]),3), "meas_op", round(float(op[t]),3), "eefz", round(float(eef[t,2]),4), "chg", int(ch[t]))
