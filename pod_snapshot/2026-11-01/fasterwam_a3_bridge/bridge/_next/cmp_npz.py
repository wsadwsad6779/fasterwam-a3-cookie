"""Compare two executed-episode logs key by key.

Used to prove that a diagnostic harness which only wraps functions with
pass-through hooks leaves the executed actions bitwise identical to the plain
inference entry point.  Reports, per shared array key, whether the two logs are
bitwise equal and, if not, the largest absolute difference.

    python3 cmp_npz.py a.npz b.npz
"""
import sys

import numpy as np


def main(a_path, b_path):
    a = np.load(a_path, allow_pickle=False)
    b = np.load(b_path, allow_pickle=False)
    ka, kb = set(a.files), set(b.files)
    rows = []
    for key in sorted(ka & kb):
        xa, xb = np.asarray(a[key]), np.asarray(b[key])
        if xa.shape != xb.shape:
            rows.append((key, str(xa.shape), str(xb.shape), "SHAPE_MISMATCH"))
            continue
        if xa.dtype.kind in "biufc":
            same = bool(np.array_equal(xa, xb))
            md = float(np.max(np.abs(xa.astype(np.float64) - xb.astype(np.float64))))
            rows.append((key, str(xa.shape), "bitwise_equal" if same else "differs", md))
        else:
            rows.append((key, str(xa.shape),
                         "bitwise_equal" if bool(np.array_equal(xa, xb)) else "differs",
                         "n/a"))
    width = max(len(r[0]) for r in rows)
    for key, shape, verdict, md in rows:
        print(f"{key:<{width}}  {shape:<16}  {verdict:<14}  max|d|={md}")
    print(f"only_in_a={sorted(ka - kb)} only_in_b={sorted(kb - ka)}")
    numeric = [r for r in rows if r[3] != "n/a"]
    print("all_numeric_bitwise_equal=" +
          str(all(r[2] == "bitwise_equal" for r in numeric)) +
          f" n_numeric={len(numeric)}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
