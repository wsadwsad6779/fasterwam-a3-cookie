import os
import sys

import numpy as np

EXP = "/workspace/expert_seed11.npz"
ROOT = "/workspace/fasterwam_a3_bridge/bridge/outputs"
STAGES = sys.argv[1:] or ["C3", "C4"]


def resample(a, n):
    a = np.asarray(a, dtype=np.float64)
    if len(a) == 0:
        return np.zeros((n,) + a.shape[1:])
    idx = np.linspace(0, len(a) - 1, n)
    lo = np.floor(idx).astype(int)
    hi = np.minimum(lo + 1, len(a) - 1)
    w = (idx - lo).reshape([-1] + [1] * (a.ndim - 1))
    return a[lo] * (1 - w) + a[hi] * w


def tos(p):
    if isinstance(p, bytes):
        return p.decode()
    return str(p)


def segments(ph):
    out = []
    for i, p in enumerate(ph):
        if not out or out[-1][0] != p:
            out.append([p, i, i + 1])
        else:
            out[-1][2] = i + 1
    return out


exp = np.load(EXP, allow_pickle=True)
ex_eef = np.asarray(exp["eef"], dtype=np.float64)[:, :3]
ex_open = np.asarray(exp["opening"], dtype=np.float64).reshape(-1)
ex_ph = [tos(p) for p in np.asarray(exp["phases"]).reshape(-1)]
ex_seg = segments(ex_ph)
print("expert steps=%d segments=%d" % (len(ex_eef), len(ex_seg)))
print("| run | k | expert_seg | run_seg | exp_n | run_n | err@25% | err@end | open_err@end |")
print("|---|---|---|---|---|---|---|---|---|")

for st in STAGES:
    d = os.path.join(ROOT, st)
    if not os.path.isdir(d):
        continue
    for f in sorted(os.listdir(d)):
        if not f.endswith(".npz"):
            continue
        r = np.load(os.path.join(d, f), allow_pickle=True)
        if "eef_xyz" in r.files:
            run_eef = np.asarray(r["eef_xyz"], dtype=np.float64)
        else:
            run_eef = np.asarray(r["eef"], dtype=np.float64)[:, :3]
        run_open = np.asarray(r["opening"], dtype=np.float64).reshape(-1)
        run_ph = [tos(p) for p in np.asarray(r["phase"]).reshape(-1)]
        run_seg = segments(run_ph)
        tag = st + "/" + f.replace(".npz", "")
        for k in range(max(len(ex_seg), len(run_seg))):
            en, es, ee = ex_seg[k] if k < len(ex_seg) else ("-", 0, 0)
            rn, rs, re_ = run_seg[k] if k < len(run_seg) else ("-", 0, 0)
            if ee <= es or re_ <= rs:
                print("| %s | %d | %s | %s | %d | %d | - | - | - |" % (tag, k, en, rn, ee - es, re_ - rs))
                continue
            ea = resample(ex_eef[es:ee], 20)
            ra = resample(run_eef[rs:re_], 20)
            err = np.linalg.norm(ra - ea, axis=1) * 1000.0
            eo = resample(ex_open[es:ee].reshape(-1, 1), 20).reshape(-1)
            ro = resample(run_open[rs:re_].reshape(-1, 1), 20).reshape(-1)
            print("| %s | %d | %s | %s | %d | %d | %.1f | %.1f | %.3f |"
                  % (tag, k, en, rn, ee - es, re_ - rs, err[5], err[-1], abs(ro[-1] - eo[-1])))
