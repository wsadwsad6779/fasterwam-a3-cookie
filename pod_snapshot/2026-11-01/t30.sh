cd /workspace
export PYTHONPATH=/workspace/a3_dual_arm_sim/src
python3 /workspace/_r4_stage/autonomy_metrics.py /workspace/a3_eval/pilot_single_lowforce_clean 2>&1 | head -20
echo ==INTARGET==
python3 - <<PY
import numpy as np
z = np.load("/workspace/a3_eval/pilot_single_lowforce_clean/expert_single_6001.npz")
it = z["in_target"]; xyz = z["cookie_xyz"]; eef = z["eef_xyz"]
print("in_target unique transitions:", [(int(i), int(v)) for i, v in zip(*np.unique(it, return_index=True))])
print("final in_target", int(it[-1]))
for c in range(2):
    p = xyz[-1, c]
    print("cookie", c, "final xyz", np.round(p, 4))
print("bin inner half 0.057 0.029 at (0.095,0.100); floor top 0.775")
print("z of cookie0 over time (every 20):", np.round(xyz[::20, 0, 2], 4))
PY
echo DONE