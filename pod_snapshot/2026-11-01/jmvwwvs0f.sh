sed -n "368,380p" /workspace/a3_dual_arm_sim/src/a3_dual_arm_sim/sim/env.py
echo ===
python3 - <<PY
import glob, numpy as np, pyarrow.parquet as pq
fs=sorted(glob.glob("/workspace/a3_data/a3-front-close-left-100/data/chunk-000/*.parquet"))[:3]
for f in fs:
    t=pq.read_table(f, columns=["observation.force"])
    a=np.asarray(t["observation.force"].to_numpy().tolist(), dtype=np.float32)
    print(f.split("/")[-1], a.shape, "max", np.abs(a).max(0).round(2).tolist())
PY