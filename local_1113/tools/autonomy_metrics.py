"""Autonomy metrics for the cookie transfer evaluation (task 4 protocol).

Reads only the recorded eval npz files, so it is applicable to both the
historical C0 protocol and the C0-raw (bare policy, no Safety rewrite) runs.
A cookie counts as genuinely transferred only if it was clamped, carried,
released, and then rests inside the target bin -- being held by a closed
gripper at the end of the episode is not a completed grasp-carry-release.
"""

import glob
import os
import sys

import numpy as np

OUT = sys.argv[1] if len(sys.argv) > 1 else "/workspace/a3_eval/r4_b0_c0_paired"

BIN_X, BIN_Y = 0.095, 0.100
HX, HY = 0.057, 0.029
FLOOR_TOP = 0.775
Z_REST = FLOOR_TOP + 0.030
CLOSED_OPENING = 0.42
RIGID_EPS = 0.002
HOLD_DIST = 0.060
TCP_MOVE = 0.002
LIFT_EPS = 0.030
EXPERT_HOLD_FORCE_N = 1.39
SOFT_N = 25.0
HARD_N = 40.0
TAIL = 50


def flips(seq):
    s = np.sign(np.diff(np.asarray(seq, dtype=np.float64)))
    s = s[s != 0]
    return int((np.diff(s) != 0).sum()) if s.size > 1 else 0


def rest_in_bin(xyz, eef, op):
    inside = (np.abs(xyz[-1, :, 0] - BIN_X) <= HX) & (np.abs(xyz[-1, :, 1] - BIN_Y) <= HY)
    near_floor = (xyz[-1, :, 2] > FLOOR_TOP - 0.004) & (xyz[-1, :, 2] < Z_REST)
    move = np.zeros((xyz.shape[0], xyz.shape[1]))
    move[1:] = np.linalg.norm(np.diff(xyz, axis=0), axis=2)
    tail_move = move[-TAIL:].max(axis=0)
    held = np.linalg.norm(xyz[-1] - eef[-1], axis=1) < HOLD_DIST
    return int(((inside & near_floor) & (tail_move < 0.008) & ~held & (op[-1] > CLOSED_OPENING)).sum())


def report(path):
    z = np.load(path)
    xyz = np.asarray(z["cookie_xyz"], dtype=np.float64)
    eef = np.asarray(z["eef_xyz"], dtype=np.float64)
    op = np.asarray(z["opening"], dtype=np.float64)
    aop = np.asarray(z["applied_opening"], dtype=np.float64) if "applied_opening" in z.files else op
    pad = np.maximum(np.asarray(z["pad_force_inner"]), np.asarray(z["pad_force_outer"]))
    wn = np.linalg.norm(np.asarray(z["wrist_force"], dtype=np.float64), axis=1)
    changed = np.asarray(z["changed"]).astype(bool) if "changed" in z.files else np.zeros(op.size, bool)
    verbatim = None
    if "verbatim_frac" in z.files:
        verbatim = float(np.asarray(z["verbatim_frac"]).ravel()[0])

    T, N = op.size, xyz.shape[1]
    rel = xyz - eef[:, None, :]
    dist = np.linalg.norm(rel, axis=2)
    rigid = np.zeros((T, N), dtype=bool)
    rigid[1:] = np.linalg.norm(np.diff(rel, axis=0), axis=2) < RIGID_EPS
    attached = rigid & (dist < HOLD_DIST) & (op[:, None] < CLOSED_OPENING)
    any_attach = attached.any(axis=1)

    events = []
    t = 0
    while t < T:
        if any_attach[t]:
            u = t
            while u + 1 < T and any_attach[u + 1]:
                u += 1
            events.append((t, u))
            t = u + 1
        else:
            t += 1

    n_single = n_bundle = n_lifted = n_not_released = n_released_in_bin = 0
    durations = []
    carries = []
    for a, b in events:
        counts = attached[a:b + 1].sum(axis=0)
        ids = np.flatnonzero(counts >= max(1, (b - a) // 3))
        n = int(attached[a:b + 1].sum(axis=1).max())
        if n >= 2:
            n_bundle += 1
        elif n == 1:
            n_single += 1
        durations.append(b - a + 1)
        if ids.size:
            z0 = xyz[a, ids, 2].mean()
            rise = float(xyz[a:b + 1][:, ids, 2].max() - z0)
            xy = xyz[a:b + 1][:, ids, 0:2]
            carries.append(float(np.linalg.norm(xy[-1].mean(axis=0) - xy[0].mean(axis=0))))
            if rise >= LIFT_EPS:
                n_lifted += 1
            tail_idx = min(T - 1, b + 10)
            still = bool(attached[tail_idx, ids].any())
            if still:
                n_not_released += 1
            else:
                cx = xyz[tail_idx, ids, 0]
                cy = xyz[tail_idx, ids, 1]
                if np.all(np.abs(cx - BIN_X) <= HX) and np.all(np.abs(cy - BIN_Y) <= HY):
                    n_released_in_bin += 1

    dur_med = float(np.median(durations)) if durations else 0.0
    car_med = float(np.median(carries)) if carries else 0.0
    name = os.path.basename(path)
    print("%-22s events=%d single=%d single_lift=%d bundle=%d held_no_release=%d rel_in_bin=%d "
          "rest_in_bin=%d dur_med=%.0f car_med=%.0fmm" % (
              name, len(events), n_single, n_lifted, n_bundle, n_not_released,
              n_released_in_bin, rest_in_bin(xyz, eef, op), dur_med, car_med * 1000))
    print("%-22s pad N: p50=%.2f p95=%.2f max=%.2f | >%.0fN %d frames, >%.0fN %d frames | "
          "wrist p50=%.2f max=%.2f" % (
              "", np.percentile(pad, 50), np.percentile(pad, 95), pad.max(),
              SOFT_N, int((pad > SOFT_N).sum()), HARD_N, int((pad > HARD_N).sum()),
              np.percentile(wn, 50), wn.max()))
    print("%-22s grip flips obs=%d cmd=%d | safety-rewritten steps=%d | cookie_in_target_end=%d" % (
        "", flips(op), flips(aop), int(changed.sum()),
        int(((np.abs(xyz[-1, :, 0] - BIN_X) <= HX) & (np.abs(xyz[-1, :, 1] - BIN_Y) <= HY)).sum())))
    if verbatim is not None:
        print("%-22s verbatim=%.4f" % ("", verbatim))
    print("")


files = sorted(glob.glob(os.path.join(OUT, "*.npz")))
print("dir=%s files=%d | expert reference hold force=%.2f N, peak=%.2f N" % (
    OUT, len(files), EXPERT_HOLD_FORCE_N, 1.54))
for f in files:
    report(f)
