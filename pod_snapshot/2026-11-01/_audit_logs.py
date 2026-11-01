import re, numpy as np
from pathlib import Path
pat=re.compile(r"step=(\d+)/\d+.{0,220}?loss=([0-9.]+).{0,120}?loss_action=([0-9.]+).{0,120}?loss_video=([0-9.]+).{0,120}?lr=([0-9.eE+-]+)", re.S)
for nm,p in [("r2","/workspace/r2.log"),("r3","/workspace/r3.log")]:
    tx=Path(p).read_text(errors="ignore")
    rows=[(int(a),float(b),float(c),float(d),float(e)) for a,b,c,d,e in pat.findall(tx)]
    print("LOG",nm,"n",len(rows),"first",rows[:2],"last",rows[-2:])
    if rows:
        arr=np.array(rows)
        for lo in range(0,int(arr[:,0].max())+1,500):
            s=arr[(arr[:,0]>=lo)&(arr[:,0]<lo+500)]
            if len(s): print("  %s %4d loss %.4f act %.4f vid %.4f lr %.2e"%(nm,lo,s[:,1].mean(),s[:,2].mean(),s[:,3].mean(),s[:,4].mean()))
    for kw in ["not restored","Resuming","resume","Loading Wan","max_steps reached"]:
        idx=[m.start() for m in re.finditer(kw,tx)]
        print("  kw",kw,len(idx),[tx[max(0,i-60):i+90].replace(chr(10)," ") for i in idx[:2]])

