import json, numpy as np
D="/workspace/a3_data/a3_fw"
print("== summary.json ==")
print(open(D+"/summary.json").read())
z=np.load(D+"/arrays.npz", mmap_mode="r")
print("== arrays.npz keys ==")
for k in z.files:
    a=z[k]
    print(f"{k:32s} {str(a.dtype):10s} {str(a.shape):18s} bytes={a.nbytes}")
print("== name search ==")
needles=["force","wrench","torque","sensor","contact","time","ts","stamp","tick","dt"]
hits=[k for k in z.files if any(n in k.lower() for n in needles)]
print("hits:", hits)
print("== episode id dtype ==")
ek=[k for k in z.files if "ep" in k.lower() or "task" in k.lower() or "lang" in k.lower()]
print(ek)
for k in ek:
    a=np.asarray(z[k])
    print(k, a.dtype, a.shape, a.reshape(-1)[:4] if a.dtype!=object else a.reshape(-1)[:2])
print("== proprio/action rows ==")
for k in z.files:
    a=np.asarray(z[k]) if np.prod(z[k].shape)<10**7 else None
    if a is not None and a.ndim>=2 and a.shape[-1] in (8,7,26):
        print(k, a.shape, "min", np.round(a.min(axis=0),4)[:a.shape[-1]], "max", np.round(a.max(axis=0),4)[:a.shape[-1]])
