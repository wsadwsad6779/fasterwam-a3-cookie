import numpy as np
A = np.load("/workspace/fasterwam_a3_bridge/bridge/outputs/E0_C0/lora_C0_11.npz", allow_pickle=True)
B = np.load("/workspace/fasterwam_a3_bridge/bridge/outputs/C5/lora_C5_11.npz", allow_pickle=True)
for k in ["eef_xyz", "opening", "pad_force_inner", "pad_force_outer", "dist_to_source"]:
    a = A[k].astype(np.float64); b = B[k].astype(np.float64)
    print(k, "max_abs_diff", round(float(np.abs(a-b).max()), 9))
am = A["applied_canonical"].astype(np.float64); bm = B["applied_canonical"].astype(np.float64)
print("applied_canonical max_abs_diff", round(float(np.abs(am-bm).max()), 9))
