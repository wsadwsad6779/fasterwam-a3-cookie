"""Verdict on the cookies that scored in the bin, from the STORED episode.

tko_mask.py replays an episode through the simulator. For the bare (B0) episode
that replay diverged (max|d_cookie_xyz| = 34 mm; 56 mask changes in the replay
against 0 in the stored run), so a replayed mask cannot be used as evidence for
B0. This script reads only the arrays run_matrix.py recorded while the episode
actually happened, so every number below is what the simulator produced.

The question it answers is the one 任务四 asks: of the cookies that scored, how
many were genuinely resting in the bin, and how many were still held by the
gripper or being dragged along in a bundle.

Scene geometry (from the same source as the authoritative scorer):
    bin centre xy (0.095, 0.100), inner half size (0.057, 0.029)
    floor top z 0.775, rim z 0.779, cookie half size (0.025, 0.00317, 0.0125)

    python3 mech_eval.py
"""
import numpy as np

BIN = np.array([0.095, 0.100])
HX, HY = 0.057, 0.029
FLOOR = 0.775
Z_REST = FLOOR + 0.030          # one upright cookie standing on the floor
OUT = "/workspace/a3_eval/r4_b0_c0_paired"
CASES = [("B0", 11), ("B0", 21), ("B0", 22), ("C0", 11), ("C0", 21), ("C0", 22)]
TAIL = 50

for ctl, seed in CASES:
    d = np.load("%s/lora_%s_%d.npz" % (OUT, ctl, seed), allow_pickle=False)
    xyz = d["cookie_xyz"].astype(float)
    eef = d["eef_xyz"].astype(float)
    opening = d["opening"].astype(float)
    cmd = d["applied_opening"].astype(float)
    pad = np.maximum(d["pad_force_inner"], d["pad_force_outer"]).astype(float)
    wrist = np.linalg.norm(d["wrist_force"].astype(float), axis=1)
    changed = d["changed"].astype(bool)
    T, N, _ = xyz.shape

    in_fp = (np.abs(xyz[:, :, 0] - BIN[0]) <= HX) & \
            (np.abs(xyz[:, :, 1] - BIN[1]) <= HY)
    near_floor = (xyz[:, :, 2] > FLOOR - 0.004) & (xyz[:, :, 2] < Z_REST)
    step_disp = np.zeros((T, N))
    step_disp[1:] = np.linalg.norm(np.diff(xyz, axis=0), axis=2)
    tail_move = step_disp[T - TAIL:].sum(axis=0)
    d3 = np.linalg.norm(xyz - eef[:, None, :], axis=2)
    rel = xyz[:, :, :2] - eef[:, None, :2]
    rigid = np.zeros((T, N), dtype=bool)
    rigid[1:] = np.linalg.norm(np.diff(rel, axis=0), axis=2) < 0.002
    # A stationary TCP makes every cookie look "rigid", so dragging requires the
    # TCP itself to be moving that step.
    tcp_moving = np.zeros(T, dtype=bool)
    tcp_moving[1:] = np.linalg.norm(np.diff(eef, axis=0), axis=1) > 0.002
    dragged = rigid & (d3 < 0.060) & tcp_moving[:, None]
    bundle = int(dragged.sum(axis=1).max())
    rest_now = in_fp[-1] & near_floor[-1] & (tail_move < 0.008)

    for i in range(N):
        if not (in_fp.any(axis=0)[i] or in_fp[-1, i] or dragged[:, i].any()):
            continue
        state = "rest" if rest_now[i] else ("HELD" if dragged[T - 1, i] else "moving")
        drag_steps = np.flatnonzero(dragged[:, i])
        pad_drag = float(pad[drag_steps].max()) if drag_steps.size else 0.0
        print("%s/%d c%-2d %-7s fp=%d z=%.4f lift=%5.1fmm xy_disp=%5.1fmm "
              "d_eef=%5.1fmm drag=%3d/%d open=%.3f pad@drag=%6.1fN"
              % (ctl, seed, i, state, int(in_fp[-1, i]), xyz[-1, i, 2],
                 1000 * (xyz[:, i, 2].max() - xyz[0, i, 2]),
                 1000 * np.linalg.norm(xyz[-1, i, :2] - xyz[0, i, :2]),
                 1000 * d3[-1, i], int(dragged[:, i].sum()), T,
                 opening[-1], pad_drag))

    above_rim = int((in_fp[-1] & (xyz[-1, :, 2] > Z_REST)).sum())
    print("-- %s seed %d: steps=%d verbatim=%.2f%% resting_in_bin=%d "
          "in_footprint_final=%d footprint_above_rest_zone=%d max_bundle=%d "
          "open_flips(obs=%d,cmd=%d) pad_peak=%.1fN@t%d wrist_peak=%.1fN@t%d"
          % (ctl, seed, T, 100.0 * (1 - changed.mean()), int(rest_now.sum()),
             int(in_fp[-1].sum()), above_rim, bundle,
             int((np.diff(np.sign(np.diff(opening))) < 0).sum()),
             int((np.diff(np.sign(np.diff(cmd))) < 0).sum()),
             pad.max(), int(pad.argmax()), wrist.max(), int(wrist.argmax())))
    print("   resting ids: %s" % [int(i) for i in range(N) if rest_now[i]])
    print("   ever in footprint: %s" % [int(i) for i in range(N) if in_fp.any(axis=0)[i]])
    print("   dragged-with-TCP ids: %s" % [int(i) for i in range(N) if dragged[:, i].sum() > T // 4])
    print("   final z sorted: %s" % np.round(np.sort(xyz[-1, :, 2])[:12], 4).tolist())
    print("")
