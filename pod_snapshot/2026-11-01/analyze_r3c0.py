import numpy as np, sys
P = "/workspace/fasterwam_a3_bridge/bridge/outputs/r3/C0/lora_C0_%d.npz"
def runs(mask, minlen=3):
    out=[]; i=0; n=len(mask)
    while i<n:
        if mask[i]:
            j=i
            while j<n and mask[j]: j+=1
            if j-i>=minlen: out.append((i,j-1,j-i))
            i=j
        else: i+=1
    return out
for s in (11,21,22):
    d=np.load(P%s); print("=== seed", s)
    pre=d["pre_14d"][:,6]; post=d["post_wrap_14d"][:,6]; postapp=d["post_14d"][:,6]
    op=d["opening"]; aop=d["applied_opening"]; z=d["eef_xyz"][:,2]
    pad=d["pad_force_inner"]+d["pad_force_outer"]
    cx=d["cookie_xyz"]; c0=d["cookie_xyz0"]
    dz=cx[:,:,2]-c0[None,:,2]; lift=dz.max(1); who=dz.argmax(1)
    dsrc=d["dist_to_source"]
    print("cmd_post runs closed(>0.5):", runs(post>0.5,5)[:8])
    print("cmd_post runs open (<0.5):", runs(post<0.5,5)[:8])
    print("pad>5:", runs(pad>5,2)[:6], " pad>10:", runs(pad>10,2)[:6], " pad>20:", runs(pad>20,2)[:4])
    k=int(np.argmax(lift)); print("lift peak %.4f @%d cookie%d  end %.4f" % (lift[k],k,who[k],lift[-1]))
    print("zmin %.3f@%d zmax %.3f@%d" % (z.min(),int(z.argmin()),z.max(),int(z.argmax())))
    print("step  cmdpost op  aop  z     lift  ck pad   dsrc")
    for t in range(0,600,10):
        print("%4d  %+5.2f  %.2f %.2f %.3f %.4f %2d %5.1f %5.2f" % (t,post[t],op[t],aop[t],z[t],lift[t],who[t],pad[t],dsrc[t]))
    for t in range(555,600):
        if s==11 and 555<=t<600 and (t%2==0 or t>=575):
            print("  fine %4d cmd %+5.2f op %.3f z %.4f lift %.4f ck%d pad %5.1f" % (t,post[t],op[t],z[t],lift[t],who[t],pad[t]))
    if s==21:
        for t in range(200,250,2):
            print("  fine %4d cmd %+5.2f op %.3f z %.4f lift %.4f ck%d pad %5.1f" % (t,post[t],op[t],z[t],lift[t],who[t],pad[t]))
    if s==22:
        for t in range(40,140,4):
            print("  fine %4d cmd %+5.2f op %.3f z %.4f lift %.4f ck%d pad %5.1f" % (t,post[t],op[t],z[t],lift[t],who[t],pad[t]))
    np.save("/workspace/r3ev_%d.npy" % s, np.stack([post,pre,op,aop,z,lift,who.astype(float),pad,dsrc],1))
