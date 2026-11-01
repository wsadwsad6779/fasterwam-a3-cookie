import json
import numpy as np

ROOT = "/workspace/fasterwam_a3_bridge/bridge/outputs/r3/C0"
CASES = [(11, 506), (11, 530), (11, 545), (21, 232), (21, 226)]

for seed, t in CASES:
    d = np.load(f"{ROOT}/lora_C0_{seed}.npz")
    xyz = np.asarray(d["cookie_xyz"], dtype=np.float64)[t]
    eef = np.asarray(d["eef_xyz"], dtype=np.float64)[t]
    op = np.asarray(d["opening"], dtype=np.float64)[t]
    floor = float(np.percentile(np.asarray(d["cookie_xyz"], dtype=np.float64)[:, :, 2], 2))
    print(f"== seed={seed} t={t} open={op:.3f} eef={np.round(eef[:3],4).tolist()}", flush=True)
    up = np.flatnonzero(xyz[:, 2] > floor + 0.008)
    for i in up.tolist():
        p = xyz[i]
        print(json.dumps({"i": i, "x": round(float(p[0]), 4), "y": round(float(p[1]), 4),
                          "z": round(float(p[2]), 4),
                          "d_eef": round(float(np.linalg.norm(p[:3]-eef[:3])), 4)}), flush=True)
