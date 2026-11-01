import numpy as np
ROOT='/workspace/fasterwam_a3_bridge/bridge/outputs/r3/C0'
d=np.load(ROOT+'/lora_C0_11.npz')
xyz=np.asarray(d['cookie_xyz'],dtype=np.float64)
xyz0=np.asarray(d['cookie_xyz0'],dtype=np.float64)
eef=np.asarray(d['eef_xyz'],dtype=np.float64)
op=np.asarray(d['opening'],dtype=np.float64)
fi=np.asarray(d['pad_force_inner'],dtype=np.float64)
fo=np.asarray(d['pad_force_outer'],dtype=np.float64)
T=len(xyz); print('T',T,'shapes',xyz.shape,xyz0.shape,fi.shape,fo.shape)
disp=np.linalg.norm(xyz-xyz0,axis=2)
for t in range(0,260,5):
    fm=float(np.max(np.abs(fi[t])))
    print(t,'eefz=%.4f'%eef[t,2],'eefxy=(%.3f,%.3f)'%(eef[t,0],eef[t,1]),'op=%.3f'%op[t],'F=%.1f'%fm,'maxdisp=%.4f'%disp[t].max(),'n>5mm=%d'%int((disp[t]>0.005).sum()),flush=True)
