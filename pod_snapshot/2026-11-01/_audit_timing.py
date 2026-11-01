import numpy as np
np.set_printoptions(precision=4, suppress=True)
zs={s:np.load(f"/workspace/fasterwam_a3_bridge/bridge/outputs/r3/C0/lora_C0_{s}.npz", allow_pickle=True) for s in (11,21,22)}
for s,z in zs.items():
    print("SEED",s,{k:(np.asarray(z[k]).shape if k in z.files else None) for k in ["cookie_xyz0","cookie_xyz","pre_14d","post_14d","changed","chunk_idx","opening","applied_canonical"]})
x0={s:np.asarray(z["cookie_xyz0"]) for s,z in zs.items()}
print("cookie_xyz0 shapes", {s:v.shape for s,v in x0.items()})
a=x0[11]; b=x0[21]; c=x0[22]
print("max diff 11-21", float(np.abs(a-b).max()) if a.shape==b.shape else "shape mismatch")
print("max diff 11-22", float(np.abs(a-c).max()) if a.shape==c.shape else "shape mismatch")
print("x0 sample", a[:3])
for s,z in zs.items():
    ch=np.asarray(z["chunk_idx"]); ap=np.asarray(z["applied_canonical"]); op=np.asarray(z["opening"]); chd=np.asarray(z["changed"])
    g=ap[:,7] if ap.shape[1]>=8 else ap[:,-1]
    starts=np.where(np.diff(ch)!=0)[0]+1
    flips=np.where(np.diff(g)!=0)[0]+1
    bf=[i for i in flips if i in set(starts)]
    print("SEED",s,"chunks",len(starts)+1,"grip flips",len(flips),"flips_at_chunk_start",len(bf),"frac",round(len(bf)/max(1,len(flips)),3),"safety_changed",int(chd.sum()),"opening min/max",float(op.min()),float(op.max()))
    print("  first 30 chunk starts g:",[(int(i),float(g[i-1]),float(g[i])) for i in starts[:12]])
    print("  opening at 220-256:", np.round(op[220:257],3))
    xy=np.asarray(z["cookie_xyz"])
    z0=np.asarray(z["cookie_xyz0"])
    if xy.ndim==3:
        lift=xy[:,:,2]-z0[None,:,2]; print("  cookie max lift mm", round(float(lift.max())*1000,2), "step", int(np.unravel_index(lift.argmax(),lift.shape)[0]))

