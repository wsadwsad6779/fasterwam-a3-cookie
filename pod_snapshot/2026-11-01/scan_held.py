import sys, json
import numpy as np

ROOT = "/workspace/fasterwam_a3_bridge/bridge/outputs/r3/C0"
WIN = [(11, 470, 600), (21, 140, 250), (22, 40, 95)]

for seed, lo, hi in WIN:
    d = np.load(f"{ROOT}/lora_C0_{seed}.npz")
    xyz = np.asarray(d["cookie_xyz"], dtype=np.float64)
    eef = np.asarray(d["eef_xyz"], dtype=np.float64)
    op = np.asarray(d["opening"], dtype=np.float64)
    floor = float(np.percentile(xyz[:, :, 2], 2))
    print(f"== seed={seed} floor={floor:.4f} T={len(xyz)}", flush=True)
    prev = None
    for t in range(lo, min(hi, len(xyz))):
        z = xyz[t, :, 2]
        up = np.flatnonzero(z > floor + 0.010)
        e = eef[t, :3]
        if len(up):
            dd = np.linalg.norm(xyz[t, up, :3] - e, axis=1)
            near = up[dd < 0.045]
        else:
            near = up
        key = (len(up), len(near))
        if key != prev:
            print(json.dumps({"t": t, "up": int(len(up)), "near": int(len(near)),
                              "near_idx": near.tolist()[:12],
                              "eef_z": round(float(e[2]), 4),
                              "eef_y": round(float(e[1]), 4),
                              "open": round(float(op[t]), 3)}), flush=True)
            prev = key
