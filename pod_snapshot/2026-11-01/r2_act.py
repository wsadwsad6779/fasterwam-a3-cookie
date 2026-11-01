import numpy as np
for s in (11,21,22):
    d=np.load('/workspace/fasterwam_a3_bridge/bridge/outputs/r2/C0/lora_C0_%d.npz'%s)
    pre=d['pre_14d']; pad=d['pad_force_inner']+d['pad_force_outer']; op=d['opening']
    c=np.where((pad>5)&(op<0.9))[0]; c0=int(c[0]) if len(c) else 0
    z=pre[:,2]
    print('seed',s,'contact',c0,'pre_z: before %.3f [%.3f,%.3f] after %.3f [+frac %.2f] last100 %.3f'%(z[:c0].mean() if c0 else 0,z[:c0].min(),z[:c0].max(),z[c0:].mean(),(z[c0:]>0.02).mean(),z[-100:].mean()))
for s in (11,21,22):
    a=np.load('/workspace/expert_seed%d.npz'%s)['act14'][:,2]
    print('expert',s,'act_z mean %.3f [+frac %.2f] max %.3f'%(a.mean(),(a>0.02).mean(),a.max()))
