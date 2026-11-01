import numpy as np, re, os
from pathlib import Path
z=np.load("/workspace/a3_data/a3_fw/arrays.npz")
A=z["action"]; P=z["proprio"]; S=z["ep_start"]; L=z["ep_len"]
from scipy.spatial.transform import Rotation as Rot
rc=[]; rr=[]; nr=[]
for s,l in zip(S,L):
    ra=Rot.from_rotvec(P[s:s+l,3:6]); dR=ra[:-1].inv()*ra[1:]; rv=dR.as_rotvec()/0.5
    a=A[s:s+l,3:6][:len(rv)]
    n=np.linalg.norm(rv,axis=1)*np.linalg.norm(a,axis=1)+1e-12
    rc.append((rv*a).sum(1)/n); rr.append(np.linalg.norm(rv-a,axis=1)/(np.linalg.norm(rv,axis=1)+1e-9)); nr.append(np.linalg.norm(rv,axis=1))
rc=np.concatenate(rc); rr=np.concatenate(rr); nr=np.concatenate(nr); m=nr>1e-6
print("ROT cos med", float(np.median(rc[m])), "relerr med", float(np.median(rr[m])), "nonzero frac", float(m.mean()), "max rv", float(nr.max()))
print("action rot range", A[:,3:6].min(0), A[:,3:6].max(0))
print("action xyz absmax", np.abs(A[:,:3]).max(0))
# log formats
t=Path("/workspace/r2.log").read_text(errors="ignore")
i=t.find("loss_action"); print("R2SNIP", repr(t[max(0,i-300):i+150]))
t3=Path("/workspace/r3.log").read_text(errors="ignore")
print("R3 step/loss counts", t3.count("step="), t3.count("loss="), "LoadingWan", t3.count("Loading Wan"))
print("R3HEAD", repr(t3[:400]))
print("R3TAIL", repr(t3[-600:]))
# parse both with flexible regex
pat=re.compile(r"step=(\d+)/\d+[^0-9]{1,120}?loss=([0-9.]+)[^0-9]{1,80}?loss_action=([0-9.]+)[^0-9]{1,80}?loss_video=([0-9.]+)[^0-9]{1,80}?lr=([0-9.eE+-]+)")
for nm,p in [("r2","/workspace/r2.log"),("r3","/workspace/r3.log")]:
    tx=Path(p).read_text(errors="ignore"); rows=[(int(a),float(b),float(c),float(d),float(e)) for a,b,c,d,e in pat.findall(tx)]
    print("LOG2", nm, len(rows), rows[:2], rows[-2:])
    if rows:
        arr=np.array(rows)
        for lo in range(0, int(arr[:,0].max())+1, 500):
            sel=arr[(arr[:,0]>=lo)&(arr[:,0]<lo+500)]
            if len(sel): print("   ",nm,lo,"loss %.4f act %.4f vid %.4f lr %.2e"%(sel[:,1].mean(),sel[:,2].mean(),sel[:,3].mean(),sel[:,4].mean()))
# source dataset
for p in ["/workspace/a3_data/a3-front-close-left-100","/workspace/a3_data"]:
    print(p, os.path.isdir(p), (os.listdir(p)[:10] if os.path.isdir(p) else None))

