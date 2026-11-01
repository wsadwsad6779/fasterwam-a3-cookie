import json
import numpy as np

ROOT = "/workspace/fasterwam_a3_bridge/bridge/outputs/r3/C0"
for seed in (11, 21):
    d = np.load(f"{ROOT}/lora_C0_{seed}.npz")
    xyz = np.asarray(d["cookie_xyz"], dtype=np.float64)
    cols = np.asarray(d["cookie_xyz0"], dtype=np.float64)[:, 0]
    fin = np.asarray(d["pad_force_inner"], dtype=np.float64)
    fout = np.asarray(d["pad_force_outer"], dtype=np.float64)
    eef = np.asarray(d["eef_xyz"], dtype=np.float64)
    op = np.asarray(d["opening"], dtype=np.float64)
    T = len(xyz)
    print(f"== seed={seed} T={T}", flush=True)
    first = None
    for t in range(T):
        dev = np.abs(xyz[t, :, 0] - cols) > 0.006
        n = int(dev.sum())
        if n > 0 and first is None:
            first = t
        if 400 <= t < 600 or (seed == 21 and 150 <= t < 260):
            if t % 10 == 0 or n == 0:
                f = float(max(fin[t], fout[t, 0] if fout.ndim > 2 else 0.0)) if fin.ndim == 1 else 0.0
                fm = float(np.max(np.abs(fin[t]))) if fin.ndim == 1 else float(np.max(np.abs(fin[t])))
                print(json.dumps({"t": t, "dev": n, "open": round(float(op[t]), 3),
                                  "force": round(fm, 2),
                                  "eef_z": round(float(eef[t, 2]), 4)}), flush=True)
    print("first_dev", first, flush=True)
