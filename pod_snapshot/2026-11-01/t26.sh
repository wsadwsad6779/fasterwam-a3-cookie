cd /workspace
grep -rnE 'slipped out of batch' -B 12 a3_dual_arm_sim/src/a3_dual_arm_sim/controllers/*.py | head -40
echo ==DRIFT==
python3 - <<PY
import numpy as np
z = np.load("/workspace/a3_eval/pilot_single_lowforce_clean/expert_single_6001.npz")
xyz, eef, op = z["cookie_xyz"], z["eef_xyz"], z["opening"]
rel = xyz - eef[:, None, :]
d = np.linalg.norm(rel, axis=2)
for c in (0, 1):
    seg = d[250:360, c]
    print("cookie", c, "dist at steps 250-360: first", round(seg[0], 4), "min", round(seg.min(), 4), "max", round(seg.max(), 4))
    dd = np.linalg.norm(np.diff(rel[:, c, :], axis=0))
    print("   max|delta rel| per step", round(float(np.max(np.linalg.norm(np.diff(rel[250:360, c], axis=0), axis=1))), 4))
print("opening range", round(float(op.min()), 3), round(float(op.max()), 3))
pf = np.maximum(z["pad_force_inner"], z["pad_force_outer"])
print("pad force during grasp2 close->fail", np.round(pf[258:360:10], 3))
PY
echo DONE