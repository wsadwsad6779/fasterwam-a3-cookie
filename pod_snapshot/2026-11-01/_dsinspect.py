import numpy as np, json
z=np.load("/workspace/a3_data/a3_fw/arrays.npz")
print("KEYS", z.files)
for k in z.files:
    a=z[k]
    print(k, a.shape, a.dtype)
print("SUMMARY", open("/workspace/a3_data/a3_fw/summary.json").read())
print("STATS", open("/workspace/a3_data/a3_fw/dataset_stats.json").read()[:1500])

