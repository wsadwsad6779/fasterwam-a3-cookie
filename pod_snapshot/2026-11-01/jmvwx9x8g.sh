cd /workspace/_r5_stage
export PYTHONPATH=/workspace/a3_dual_arm_sim/src
python3 - <<EOF
import numpy as np
z=np.load("/workspace/a3_eval/demos_correct_c2/expert_correct_7244.npz")
print(sorted(z.files))
print([int(x) for x in z["phase"]][:5], z["cookie_xyz0"].shape if "cookie_xyz0" in z.files else None)
EOF
echo == FIN1
cat /workspace/fin1.out 2>/dev/null | tail -5
ps -eo pid,etimes,cmd | grep -E replay_ | grep -v grep | wc -l