import numpy as np
ROOT='/workspace/fasterwam_a3_bridge/bridge/outputs/r3/C0'
for seed in (11,21,22):
    d=np.load(ROOT+f'/lora_C0_{seed}.npz')
    xyz=np.asarray(d['cookie_xyz'],dtype=np.float64); xyz0=np.asarray(d['cookie_xyz0'],dtype=np.float64)
    op=np.asarray(d['opening'],dtype=np.float64)
    dz=xyz[:,:,2]-xyz0[:,2]
    T=len(op)
    print('== seed',seed,'T',T,flush=True)
    events=[]
    for t in range(2,T-1):
        lift=int((dz[t]>0.005).sum())
        d3=op[t+1]-op[t-1]
        if lift>=2 and d3>0.05:
            events.append((t,lift,round(float(op[t]),3),round(float(d3),3)))
    # compress consecutive
    out=[]
    for e in events:
        if not out or e[0]-out[-1][-1][0]>10: out.append([e])
        else: out[-1].append(e)
    for grp in out[:12]:
        print('event',grp[0],'->',grp[-1],flush=True)
