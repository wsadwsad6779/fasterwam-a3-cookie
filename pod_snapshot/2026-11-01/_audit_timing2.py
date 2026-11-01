import numpy as np
zs={s:np.load(f"/workspace/fasterwam_a3_bridge/bridge/outputs/r3/C0/lora_C0_{s}.npz", allow_pickle=True) for s in (11,21,22)}
for s,z in zs.items():
    pre=np.asarray(z["pre_14d"]); ch=np.asarray(z["chunk_idx"]); op=np.asarray(z["opening"])
    g=pre[:,6]
    starts=set((np.where(np.diff(ch)!=0)[0]+1).tolist())
    flips=np.where(np.diff(g)!=0)[0]+1
    bf=[i for i in flips if i in starts]
    runs=1+int((np.diff(g)!=0).sum())
    print("SEED",s,"g uniq",np.unique(g)[:6],"sign flips",len(flips),"runs",runs,"chunks",len(starts)+1,"flips_at_start",len(bf),"pct_of_flips",round(100*len(bf)/max(1,len(flips)),1))
    print("  label 218-258:", g[218:259].astype(int).tolist())
    print("  opening 218-258:", np.round(op[218:259],2).tolist())
    print("  chunk start idx:", sorted(starts)[:8], "...", sorted(starts)[-3:])

