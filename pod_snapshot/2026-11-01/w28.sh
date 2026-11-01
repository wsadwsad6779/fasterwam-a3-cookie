python3 - <<'PYEOF'
import numpy as np
z = np.load("/workspace/a3_eval/demos_correct_c1/expert_correct_7228.npz")
pad = np.maximum(z["pad_force_inner"], z["pad_force_outer"])
idx = np.flatnonzero(pad[451:] > 2.0) + 451
print("over2 indices", idx.tolist())
print("window 446..462", np.round(pad[446:462], 3).tolist())
print("opening window 446..462", np.round(z["opening"][446:462], 3).tolist())
print("phase window 446..462", z["phase"][446:462].tolist())
PYEOF
date
grep -c . /workspace/cc1.out
echo __END__
