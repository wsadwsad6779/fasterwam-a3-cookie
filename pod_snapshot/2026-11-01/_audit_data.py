import numpy as np, json, re
from pathlib import Path
np.set_printoptions(precision=4, suppress=True)
z=np.load("/workspace/a3_data/a3_fw/arrays.npz")
A=z["action"]; P=z["proprio"]; O=z["opening"]; J=z["joint_cmd"]; S=z["ep_start"]; L=z["ep_len"]; ID=z["ep_ids"]
print("SUM len", L.sum(), "S0", S[0], "Slast+L", S[-1]+L[-1])
print("contiguous", bool(np.all(S[1:]==S[:-1]+L[:-1])))
win=int(((L-32)).sum()); print("WINDOWS", win, "formula", L.sum()-32*len(L))
print("ep_len min/med/max", L.min(), int(np.median(L)), L.max())
print("joint_cmd uniq", np.unique(J)[:10], "counts", [(v,int((J==v).sum())) for v in np.unique(J)][:6])
print("action grip uniq", np.unique(A[:,6])[:10], "mean", A[:,6].mean())
print("opening min/med/max", O.min(), np.median(O), O.max())
print("proprio grip consistency", float(np.abs(P[:,6]-0.04*O).max()), float(np.abs(P[:,7]+0.04*O).max()))
# time alignment: action xyz vs diff of proprio
cos=[]; rmse=[]; rel=[]
for s,l in zip(S,L):
    p=P[s:s+l,:3]; a=A[s:s+l,:3]
    d=np.diff(p,axis=0)/0.05
    d=d[:len(a)-1]; aa=a[:len(a)-1]
    n=np.linalg.norm(d,axis=1)*np.linalg.norm(aa,axis=1)+1e-9
    cos.append((d*aa).sum(1)/n); rmse.append(np.sqrt(((d-aa)**2).mean(1))); rel.append(np.linalg.norm(d-aa,axis=1)/(np.linalg.norm(d,axis=1)+1e-6))
cos=np.concatenate(cos); rmse=np.concatenate(rmse); rel=np.concatenate(rel)
m=np.linalg.norm(np.concatenate([np.diff(P[s:s+l,:3],axis=0) for s,l in zip(S,L)]),axis=1)>1e-6
print("xyz align cos med/10pct", float(np.median(cos[m])), float(np.percentile(cos[m],10)), "rmse med", float(np.median(rmse[m])), "relerr med", float(np.median(rel[m])))
# gripper label vs measured opening change
ag=[]; tot=0; near0=0; maxo=0
for s,l in zip(S,L):
    o=O[s:s+l]; lab=A[s:s+l,6]
    do=np.diff(o); lab2=lab[:len(do)]
    valid=(o[:-1]>0.002)|(o[1:]>0.002)
    agree=((lab2>0)&(do<0))|((lab2<0)&(do>0))
    ag.append(agree[valid]); tot+=int(valid.sum())
    near0+=int((o<0.002).sum()); maxo+=int((o>0.998).sum())
ag=np.concatenate(ag)
print("gripper label agree rate", float(ag.mean()), "n", int(ag.size), "closed_frac", near0/int(L.sum()), "open_frac", maxo/int(L.sum()))
# start pose coverage
starts=np.stack([P[s,:3] for s in S]); ends=np.stack([P[s+l-1,:3] for s,l in zip(S,L)])
print("start xyz mean", starts.mean(0), "std", starts.std(0))
print("start xyz min", starts.min(0), "max", starts.max(0))
D=np.linalg.norm(starts[:,None,:]-starts[None,:,:],axis=2); np.fill_diagonal(D,1e9)
print("nn start dist med/min", float(np.median(D.min(1))), float(D.min(1).min()))
print("end xyz mean", ends.mean(0), "std", ends.std(0))
print("ep duration sec mean/min/max", L.mean()/20, L.min()/20, L.max()/20)
# frame count
import os
fs=os.path.getsize("/workspace/a3_data/a3_fw/frames.dat"); print("frames bytes", fs, "frames", fs/(224*448*3))
# log parse
pat=re.compile(r"step=(\d+)/\d+\s+loss=([\d.]+)\s+loss_action=([\d.]+)\s+loss_video=([\d.]+)\s+lr=([\d.eE+-]+)", re.S)
for nm,p in [("r1","/workspace/fasterwam-a3-cookie/finetune/logs/a3_lora_r1.log"),("r2","/workspace/r2.log"),("r3","/workspace/r3.log")]:
    t=Path(p).read_text(errors="ignore")
    rows=[(int(a),float(b),float(c),float(d),float(e)) for a,b,c,d,e in pat.findall(t)]
    print("LOG", nm, "n", len(rows), "maxstep", rows[-1][0] if rows else None, "restarts", t.count("Loading Wan"))
    if rows:
        arr=np.array(rows)
        for lo in range(0, arr[:,0].max().astype(int)+1, 500):
            sel=arr[(arr[:,0]>=lo)&(arr[:,0]<lo+500)]
            if len(sel): print("   ", nm, lo, "-", lo+499, "n",len(sel), "loss %.4f act %.4f vid %.4f lr %.2e"%(sel[:,1].mean(),sel[:,2].mean(),sel[:,3].mean(),sel[:,4].mean()))

