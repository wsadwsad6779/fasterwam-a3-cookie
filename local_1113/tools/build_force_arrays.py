"""Build the V1 training arrays: a3_fw arrays plus the per-frame 18-d force channel.

Reads only; writes a NEW directory. Nothing under /workspace/a3_data/a3_fw is touched.
"""
import json
import pathlib

import numpy as np
import pyarrow.parquet as pq

CORPUS = pathlib.Path("/workspace/data/a3_front_close_left_100")
BASE = pathlib.Path("/workspace/a3_data/a3_fw/arrays.npz")
OUT = pathlib.Path("/workspace/a3_data/a3_fw_force")

FORCE_V6 = [0, 1, 2, 12, 13, 16]
V6_NAMES = ["L_wrist_fx", "L_wrist_fy", "L_wrist_fz", "L_pad_inner", "L_pad_outer", "L_gripper_act"]


def main():
    base = np.load(BASE)
    n = int(base["proprio"].shape[0])
    files = sorted(CORPUS.glob("data/*/*.parquet"))
    rows = []
    for path in files:
        table = pq.read_table(
            path,
            columns=["episode_index", "frame_index", "timestamp",
                     "observation.state", "observation.force"],
        )
        for rec in table.to_pylist():
            rows.append((rec["episode_index"], rec["frame_index"], rec["timestamp"],
                         rec["observation.state"], rec["observation.force"]))
    if len(rows) != n:
        raise SystemExit("corpus has %d rows, arrays.npz has %d" % (len(rows), n))
    ep = np.asarray([r[0] for r in rows], dtype=np.int64)
    frame = np.asarray([r[1] for r in rows], dtype=np.int64)
    ts = np.asarray([r[2] for r in rows], dtype=np.float64)
    state = np.asarray([r[3] for r in rows], dtype=np.float64)
    force = np.asarray([r[4] for r in rows], dtype=np.float32)

    if not np.array_equal(ep, np.asarray(base["episode_index"])):
        raise SystemExit("episode_index differs from arrays.npz; row order is not the same")
    gap = np.abs(state[:, 7] - np.asarray(base["opening"], dtype=np.float64))
    print("rows", n, "files", len(files))
    print("state7_vs_opening_max_abs_diff", float(gap.max()))
    if gap.max() > 1e-6:
        raise SystemExit("force rows are not frame-aligned with the training arrays")
    prop = np.abs(state[:, :8] - np.asarray(base["proprio"], dtype=np.float64))
    print("state8_vs_proprio_max_abs_diff", float(prop.max()))

    starts = np.flatnonzero(np.r_[True, ep[1:] != ep[:-1]])
    lens = np.diff(np.r_[starts, n])
    ep_ids = np.asarray(sorted(set(ep.tolist())), dtype=np.int64)
    print("episodes", len(starts), "lens min/med/max",
          int(lens.min()), int(np.median(lens)), int(lens.max()))

    zero = np.zeros((len(ep_ids), force.shape[1]), dtype=np.float64)
    for k, st in enumerate(starts):
        head = force[st:st + min(20, int(lens[k]))].astype(np.float64)
        zero[k] = np.median(head, axis=0)
    sub = force[:, FORCE_V6].astype(np.float64)
    sub_zero = zero[:, FORCE_V6]
    residual = sub - sub_zero[ep]
    stats = {
        "channels": [int(c) for c in FORCE_V6],
        "names": V6_NAMES,
        "unit": "N",
        "hz": 20,
        "zero_load_ref_per_episode": sub_zero.tolist(),
        "zero_load_ref_median": np.median(sub_zero, axis=0).tolist(),
        "raw_median": np.median(sub, axis=0).tolist(),
        "raw_p95": np.percentile(sub, 95, axis=0).tolist(),
        "raw_max": sub.max(axis=0).tolist(),
        "residual_median": np.median(residual, axis=0).tolist(),
        "residual_p95": np.percentile(residual, 95, axis=0).tolist(),
        "residual_absmax": np.abs(residual).max(axis=0).tolist(),
        "residual_mean": residual.mean(axis=0).tolist(),
        "residual_std": residual.std(axis=0).tolist(),
    }
    if OUT.exists() and any(OUT.glob("*")):
        raise SystemExit("refusing to write into a non-empty directory: " + str(OUT))
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT / "arrays.npz",
        action=np.asarray(base["action"]),
        proprio=np.asarray(base["proprio"]),
        opening=np.asarray(base["opening"]),
        joint_cmd=np.asarray(base["joint_cmd"]),
        episode_index=ep,
        ep_start=starts.astype(np.int64),
        ep_len=lens.astype(np.int64),
        ep_ids=ep_ids,
        force18=force,
        force6=sub.astype(np.float32),
        force6_zeroed=(residual).astype(np.float32),
        timestamp=ts.astype(np.float32),
        frame_index=frame,
    )
    (OUT / "force_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print("wrote", OUT / "arrays.npz")
    print("names", stats["names"])
    for k in ("raw_median", "raw_p95", "raw_max", "zero_load_ref_median",
              "residual_median", "residual_p95", "residual_absmax", "residual_mean",
              "residual_std"):
        print(k, [round(float(x), 4) for x in stats[k]])


main()
