import glob, json, numpy as np
f = sorted(glob.glob("/workspace/a3_eval/r4_b0_c0_paired/**/*.npz", recursive=True))
print("files", len(f))
z = np.load(f[0])
for k in z.files:
    a = z[k]
    print(k.ljust(26), str(a.dtype).ljust(8), a.shape)
print("--- manifest ---")
print(json.dumps(json.load(open("/workspace/a3_eval/r4_b0_c0_paired/manifest_lora.json")), indent=1)[:1200])
