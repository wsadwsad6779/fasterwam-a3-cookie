ls /workspace/a3_eval/r4_b0_c0_paired/
python - <<'PY'
import glob, numpy as np
f = sorted(glob.glob("/workspace/a3_eval/r4_b0_c0_paired/**/*.npz", recursive=True))
print(len(f)); print("
".join(f[:8]))
z = np.load(f[0])
for k in z.files:
    a = z[k]
    print(k, a.dtype, a.shape)
PY
