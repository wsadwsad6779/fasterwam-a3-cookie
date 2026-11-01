import json
import numpy as np

OLD = "/workspace/a3_eval/r4_c0_frozen/lora_C0_22.npz"
NEW = "/workspace/a3_eval/r4_c0_seed22_video/lora_C0_22.npz"
a = np.load(OLD, allow_pickle=False)
b = np.load(NEW, allow_pickle=False)
print("key_sets_identical", sorted(a.files) == sorted(b.files))
all_eq = True
for k in sorted(a.files):
    xa, xb = a[k], b[k]
    if xa.shape != xb.shape:
        print(k, "SHAPE_MISMATCH", xa.shape, xb.shape)
        all_eq = False
        continue
    eq = bool(np.array_equal(xa, xb))
    d = 0.0 if eq else float(np.max(np.abs(xa.astype(float) - xb.astype(float))))
    print(k, xa.shape, "bitwise_equal", eq, "max_abs_diff", round(d, 12))
    all_eq = all_eq and eq
print("ALL_FIELDS_BITWISE_EQUAL", all_eq)


def pick(path, seed):
    for r in json.load(open(path)):
        if r["seed"] == seed:
            return r
    return {}


old_s = pick("/workspace/a3_eval/r4_c0_frozen/summary_lora.json", 22)
new_s = pick("/workspace/a3_eval/r4_c0_seed22_video/summary_lora.json", 22)
diff_keys = []
for k in sorted(old_s):
    same = old_s[k] == new_s.get(k)
    if not same:
        diff_keys.append(k)
    print(("SUM " + k).ljust(32), "old=", old_s[k], "new=", new_s.get(k), "same", same)
print("SUMMARY_DIFF_FIELDS", diff_keys)
