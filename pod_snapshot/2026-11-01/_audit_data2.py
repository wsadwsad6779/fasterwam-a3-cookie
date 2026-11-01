import numpy as np, re, os, json
from pathlib import Path
np.set_printoptions(precision=4, suppress=True)
z=np.load("/workspace/a3_data/a3_fw/arrays.npz")
A=z["action"]; P=z["proprio"]; O=z["opening"]; J=z["joint_cmd"]; S=z["ep_start"]; L=z["ep_len"]
print("joint_cmd min/max/med", float(J.min()), float(J.max()), float(np.median(J)))
print("joint_cmd vs opening corr", float(np.corrcoef(J,O)[0,1]))
print("opening pctl", [float(np.percentile(O,q)) for q in (0,1,5,25,50,75,95,99,100)])
print("grip label frac close", float((A[:,6]>0).mean()))
# refined gripper agreement: exclude holding
res={"hold":0,"move":0,"agree":0,"close_but_open_up":0,"open_but_close":0,"close_while_open":0,"open_while_close":0}
thr=0.002
for s,l in zip(S,L):
    o=O[s:s+l]; lab=A[s:s+l,6]; do=np.diff(o); l2=lab[:len(do)]
    hold=np.abs(do)<=thr; mv=~hold
    res["hold"]+=int(hold.sum()); res["move"]+=int(mv.sum())
    res["agree"]+=int((((l2>0)&(do<-thr))|((l2<0)&(do>thr)))[mv].sum())
    res["close_but_open_up"]+=int(((l2>0)&(do>thr))[mv].sum())
    res["open_but_close"]+=int(((l2<0)&(do<-thr))[mv].sum())
    res["close_while_open"]+=int(((l2>0)&(o[:-1]>0.9))[mv].sum())
    res["open_while_close"]+=int(((l2<0)&(o[:-1]<0.35))[mv].sum())
print("GRIP refined", res, "move_agree", res["agree"]/max(1,res["move"]))
# joint_cmd sign vs label
dj=[]; ag2=[]
for s,l in zip(S,L):
    j=J[s:s+l]; lab=A[s:s+l,6]; dj=np.diff(j); l2=lab[:len(dj)]
    m=np.abs(dj)>1e-5
    if m.any(): ag2.append((((l2>0)&(dj<0))|((l2<0)&(dj>0)))[m])
ag2=np.concatenate(ag2); print("label vs joint_cmd move agree", float(ag2.mean()), "n", int(ag2.size))
# rotation alignment
from scipy.spatial.transform import Rotation as R
rc=[]; rr=[]
for s,l in zip(S,L):
    ra=R.from_rotvec(P[s:s+l,3:6]); dR=ra[:-1].inv()*ra[1:]; rv=dR.as_rotvec()/0.5
    a=A[s:s+l,3:6][:len(rv)]
    n=np.linalg.norm(rv,axis=1)*np.linalg.norm(a,axis=1)+1e-12
    rc.append((rv*a).sum(1)/n); rr.append(np.linalg.norm(rv-a,axis=1)/(np.linalg.norm(rv,axis=1)+1e-9))
rc=np.concatenate(rc); rr=np.concatenate(rr); m=np.linalg.norm(A[:,3:6],axis=1)>1e-6
print("rot align cos med", float(np.median(rc[m])), "relerr med", float(np.median(rr[m])), "frac_nonzero", float(m.mean()))
# log debug
t=Path("/workspace/r2.log").read_text(errors="ignore")
i=t.find("loss_action"); print("R2 snippet", repr(t[i-260:i+120]))
t3=Path("/workspace/r3.log").read_text(errors="ignore")
print("R3 has step", t3.count("step="), "loss", t3.count("loss="), "lines", t3.count(chr(10)))
print("R3 head snippet", repr(t3[:600]))
print("R3 tail snippet", repr(t3[-800:]))
# source dirs
for p in ["/workspace/a3_data/a3-front-close-left-100", "/workspace/a3_data"]:
    print(p, os.path.isdir(p), os.listdir(p)[:8] if os.path.isdir(p) else None)

