import numpy as np, glob, json
fs=sorted(glob.glob("/workspace/fasterwam_a3_bridge/bridge/outputs/r3/C0/lora_C0_*.npz"))
print("files",fs)
for f in fs:
    z=np.load(f, allow_pickle=True)
    print(f.split("/")[-1], "keys", list(z.files)[:14])
shapes={}
for f in fs:
    z=np.load(f, allow_pickle=True)
    for k in ["oracle_cookie","oracle_data","oracle_sensor"]:
        if k in z.files: print(k, f.split("_")[-1], np.asarray(z[k]).shape, np.asarray(z[k]).dtype)
print(open("/workspace/r3_c0_metrics.txt").read())

