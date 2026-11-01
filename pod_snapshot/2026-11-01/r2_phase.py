import numpy as np
for s in (11,21,22):
    d=np.load('/workspace/fasterwam_a3_bridge/bridge/outputs/r2/C0/lora_C0_%d.npz'%s)
    eef=d['eef_xyz']; z=eef[:,2]; op=d['opening']; pad=d['pad_force_inner']+d['pad_force_outer']; n=len(z)
    ci=np.where(op<0.9)[0]; cz=int(ci[0]) if len(ci) else -1
    cm=np.where(op<0.1)[0]; cmi=int(cm[0]) if len(cm) else -1
    kz=int(np.argmin(z))
    print('seed',s,'first_close(op<.9)@',cz,'z=%.3f'%(z[cz] if cz>=0 else -1),'full_close@',cmi,'z=%.3f'%(z[cmi] if cmi>=0 else -1),'zmin %.3f@%d'%(z[kz],kz))
    for a in range(0,n,100):
        b=min(a+100,n); print('  %3d-%3d z %.3f op %.2f pad %.1f'%(a,b,z[a:b].mean(),op[a:b].mean(),pad[a:b].mean()))
