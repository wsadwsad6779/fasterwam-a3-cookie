python3 - <<PY
import numpy as np
a=np.load("/workspace/a3_data/a3_fw_force/arrays.npz")
print(sorted(a.files))
print([a[k].shape for k in ("force18","force6","proprio","action")])
PY
echo ===
grep -c . /dev/null; ls /workspace/a3_train_runs | head -12