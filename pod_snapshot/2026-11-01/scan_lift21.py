import numpy as np
d=np.load('/workspace/fasterwam_a3_bridge/bridge/outputs/r3/C0/lora_C0_21.npz')
xyz=np.asarray(d['cookie_xyz'],dtype=np.float64); xyz0=np.asarray(d['cookie_xyz0'],dtype=np.float64)
eef=np.asarray(d['eef_xyz'],dtype=np.float64); op=np.asarray(d['opening'],dtype=np.float64)
dz=xyz[:,:,2]-xyz0[:,2]
for t in (227,274,406,484,514):
    idx=[int(i) for i in np.argsort(-dz[t]) if dz[t,i]>0.005]
    dist=np.linalg.norm(xyz[t,idx,:]-eef[t],axis=-1) if idx else []
    print(t,'lifted',idx,'heights',[round(float(dz[t,i]),3) for i in idx],'dist_mm',[round(float(v)*1000,1) for v in dist],'op',round(float(op[t]),3),'eefz',round(float(eef[t,2]),4),flush=True)
