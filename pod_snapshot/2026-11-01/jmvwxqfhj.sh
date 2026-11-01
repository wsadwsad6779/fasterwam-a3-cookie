cd /workspace/_r5_stage
python3 - <<EOF
import numpy as np, json
z = np.load("/workspace/a3_data/a3_fw_force/arrays.npz")
print(sorted(z.files))
f = np.asarray(z["force18"]); epi = np.asarray(z["episode_index"])
print("shape", f.shape, "episodes", int(epi.max()) + 1)
sel = [0, 1, 2, 12, 13, 16]
names = ["L_wrist_fx", "L_wrist_fy", "L_wrist_fz", "L_pad_inner", "L_pad_outer", "L_gripper_act"]
ref = np.zeros((int(epi.max()) + 1, 6))
for k in range(int(epi.max()) + 1):
    m = epi == k
    seg = f[m][:20][:, sel]
    ref[k] = np.median(seg, axis=0)
print("per-episode zero-load reference (first 20 frames):")
for j, nm in enumerate(names):
    c = ref[:, j]
    print("  %-14s median %9.4f  mean %9.4f  std %7.4f  min %9.4f  max %9.4f  p95_abs_dev %7.4f" % (nm, np.median(c), c.mean(), c.std(), c.min(), c.max(), np.percentile(np.abs(c - np.median(c)), 95)))
st = json.load(open("/workspace/a3_data/a3_fw_force/force_stats.json"))
print("stats_keys", sorted(st.keys())[:12])
print(json.dumps(st, indent=0)[:900])
EOF