import numpy as np, sys, json
p = "/workspace/fasterwam_a3_bridge/bridge/outputs/r3/C0/lora_C0_%s.npz" % sys.argv[1]
z = np.load(p)
out = {}
for k in z.files:
    a = z[k]
    out[k] = [list(a.shape), str(a.dtype)]
print(json.dumps(out, indent=1))
