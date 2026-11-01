import numpy as np
D='/workspace/fasterwam_a3_bridge/bridge/outputs/r2/C0/lora_C0_%d.npz'
for s in (11,21,22):
    d=np.load(D%s); eef=d['eef_xyz']; op=d['opening']; pad=d['pad_force_inner']+d['pad_force_outer']
    z=eef[:,2]; n=len(z)
    c=np.where((pad>5)&(op<0.9))[0]
    c0=int(c[0]) if len(c) else -1
    k=int(np.argmax(z))
    x=np.load('/workspace/expert_seed%d.npz'%s)['eef']
    tgt=x[-1,:3]; dsrc=d['dist_to_source']
    print('seed',s,'n',n,'contact',c0,'z0 %.3f zmax %.3f@%d zend %.3f'%(z[0],z[k],k,z[-1]))
    if c0>=0:
        seg=slice(c0,n); print('  after-contact: pad_mean %.1f pad_end100 %.1f fracpad>5 %.2f op_min %.3f op_end %.3f'%(pad[seg].mean(),pad[-100:].mean(),(pad[seg]>5).mean(),op[seg].min(),op[-1]))
    print('  end eef',np.round(eef[-1],3),'expert_end',np.round(tgt,3),'dist_end %.1f mm'%(1000*np.linalg.norm(eef[-1]-tgt)))
    print('  dist_to_source: start %.3f min %.3f end %.3f'%(dsrc[0],dsrc.min(),dsrc[-1]))
