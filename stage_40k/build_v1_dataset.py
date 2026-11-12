"""Build the V1 force-aware training directory: 14-wide proprio, no processor change.

Usage: build_v1_dataset.py <out_dir> <zero_mode>
  zero_mode = episode | global

Reads a3_fw + a3_fw_force, writes a NEW directory. Nothing existing is modified.
frames.dat is symlinked, not copied (12.8 GB).

episode: subtract each episode's own median of the first min(20, ep_len) frames.
         This is what a3_fw_force/force6_zeroed already stores. REJECTED for
         training: an inference rollout only has k force samples at its k-th step
         and replans every 5 steps, so the running median deviates from the
         20-frame median by up to 5.0 residual_std (measured, k=6, wrist_fy).
global:  subtract the fixed zero_load_ref_median vector. Stateless, so training
         and inference apply the identical operation by construction. The cost
         is that per-episode load-cell zero drift is not removed; that drift is
         bounded by 0.22 residual_std across the 100 episodes (reported below).

The 6 force columns are normalised by the residual std of the chosen mode. That
division is expressed through the stock min/max normaliser by writing
global_min=-std, global_max=+std, which yields exactly scale=1/std, offset=0;
SingleFieldLinearNormalizer.forward already clips to [-5,5].
"""
import json
import pathlib
import sys

import numpy as np

BASE = pathlib.Path("/workspace/a3_data/a3_fw")
FORCE = pathlib.Path("/workspace/a3_data/a3_fw_force")

FORCE_V6 = [0, 1, 2, 12, 13, 16]
V6_NAMES = ["L_wrist_fx", "L_wrist_fy", "L_wrist_fz",
            "L_pad_inner", "L_pad_outer", "L_gripper_act"]
STAT_KINDS = ["global_min", "global_max", "global_mean", "global_std",
              "global_q01", "global_q99"]


def die(msg):
    print("FATAL: " + msg, flush=True)
    sys.exit(1)


def main():
    if len(sys.argv) != 3:
        die("usage: build_v1_dataset.py <out_dir> <episode|global>")
    out = pathlib.Path(sys.argv[1])
    mode = sys.argv[2]
    if mode not in ("episode", "global"):
        die("unknown zero_mode " + mode)
    if out.exists() and any(out.glob("*")):
        die("refusing to write into a non-empty directory: " + str(out))
    print("out", str(out), "zero_mode", mode)

    base = np.load(BASE / "arrays.npz", allow_pickle=True)
    forc = np.load(FORCE / "arrays.npz", allow_pickle=True)
    fstats = json.loads((FORCE / "force_stats.json").read_text())
    bstats = json.loads((BASE / "dataset_stats.json").read_text())

    print("base keys", sorted(base.files))
    bp = np.asarray(base["proprio"])
    fp = np.asarray(forc["proprio"])
    n = int(bp.shape[0])
    if int(fp.shape[0]) != n:
        die("row count differs: base %d force %d" % (n, fp.shape[0]))
    if bp.shape[1] != 8:
        die("base proprio is %d wide, expected 8" % bp.shape[1])
    for key in ("episode_index", "ep_start", "ep_len", "ep_ids", "action", "opening", "joint_cmd"):
        if not np.array_equal(np.asarray(base[key]), np.asarray(forc[key])):
            die("base and force disagree on " + key)
    d8 = np.abs(bp.astype(np.float64) - fp.astype(np.float64)).max()
    print("base_vs_force_proprio8_max_abs_diff", float(d8))
    if d8 > 1e-6:
        die("force arrays were built from a different proprio")

    if list(fstats["channels"]) != FORCE_V6:
        die("force_stats channels %s != %s" % (fstats["channels"], FORCE_V6))
    f18 = np.asarray(forc["force18"], dtype=np.float64)
    raw6 = np.asarray(forc["force6"], dtype=np.float64)
    stored_zeroed = np.asarray(forc["force6_zeroed"], dtype=np.float64)
    if not np.allclose(raw6, f18[:, FORCE_V6], atol=1e-5, rtol=0):
        die("force6 is not force18[:, FORCE_V6]")

    starts = np.asarray(forc["ep_start"], dtype=np.int64)
    lens = np.asarray(forc["ep_len"], dtype=np.int64)
    ep = np.asarray(forc["episode_index"], dtype=np.int64)
    n_ep = len(starts)
    print("rows", n, "episodes", n_ep, "lens min/med/max",
          int(lens.min()), int(np.median(lens)), int(lens.max()))

    ep_zero = np.zeros((n_ep, 6), dtype=np.float64)
    for k, st in enumerate(starts):
        ep_zero[k] = np.median(f18[st:st + min(20, int(lens[k]))][:, FORCE_V6], axis=0)
    drecon = np.abs((raw6 - ep_zero[ep]) - stored_zeroed).max()
    print("stored_force6_zeroed_reconstruction_max_abs_diff", float(drecon))
    if drecon > 2e-4:
        die("stored force6_zeroed does not match per-episode zeroing")

    ep_std = np.asarray(fstats["residual_std"], dtype=np.float64)
    global_zero = np.asarray(fstats["zero_load_ref_median"], dtype=np.float64)
    if global_zero.shape != (6,):
        die("zero_load_ref_median shape %s" % (global_zero.shape,))

    # How much per-episode zero drift the global reference leaves behind.
    drift = np.abs(ep_zero - global_zero).max(axis=0) / ep_std
    print("per_episode_zero_drift_vs_global_in_episode_std_units",
          [round(float(x), 5) for x in drift])
    print("worst_drift_episode", int(np.argmax(np.abs(ep_zero - global_zero).max(axis=1))))

    if mode == "episode":
        zeroed = raw6 - ep_zero[ep]
        ref_desc = "per-episode median of first min(20, ep_len) frames"
        ref_value = ep_zero.tolist()
    else:
        zeroed = raw6 - global_zero
        ref_desc = "fixed global zero_load_ref_median"
        ref_value = global_zero.tolist()

    std = zeroed.std(axis=0)
    mean = zeroed.mean(axis=0)
    if std.shape != (6,):
        die("std shape %s" % (std.shape,))
    if (std < 1e-3).any():
        die("a force std is below 1e-3; min/max ignore-dim (range_tol 1e-4) would distort it")
    print("residual_std", [round(float(x), 6) for x in std])
    print("residual_mean", [round(float(x), 6) for x in mean])
    print("residual_absmax", [round(float(x), 6) for x in np.abs(zeroed).max(axis=0)])

    proprio14 = np.concatenate([bp.astype(np.float32), zeroed.astype(np.float32)], axis=1)
    if proprio14.shape != (n, 14):
        die("proprio14 shape %s" % (proprio14.shape,))
    if not np.isfinite(proprio14).all():
        die("proprio14 contains non-finite values")

    # The normaliser reads float32 stats, so verify the trick at that precision.
    std32 = std.astype(np.float32).astype(np.float64)
    action_stats = bstats["action"]["default"]
    state8 = bstats["state"]["default"]
    for kind in STAT_KINDS:
        if len(state8[kind]) != 8:
            die("base state %s is %d wide" % (kind, len(state8[kind])))
    state14 = {
        "global_min": list(state8["global_min"]) + (-std32).tolist(),
        "global_max": list(state8["global_max"]) + std32.tolist(),
        "global_mean": list(state8["global_mean"]) + mean.tolist(),
        "global_std": list(state8["global_std"]) + std.tolist(),
        "global_q01": list(state8["global_q01"]) + np.percentile(zeroed, 1, axis=0).tolist(),
        "global_q99": list(state8["global_q99"]) + np.percentile(zeroed, 99, axis=0).tolist(),
    }

    scale = 2.0 / (np.asarray(state14["global_max"]) - np.asarray(state14["global_min"]))
    offset = -1.0 - scale * np.asarray(state14["global_min"])
    print("force_scale", [round(float(x), 6) for x in scale[8:]])
    print("force_offset", [float(x) for x in offset[8:]])
    if not np.allclose(scale[8:], 1.0 / std32, rtol=1e-6, atol=0):
        die("min/max trick did not yield scale=1/std")
    if not np.allclose(offset[8:], 0.0, atol=1e-6):
        die("min/max trick did not yield offset=0")
    pre_clip = zeroed * scale[8:] + offset[8:]
    norm_force = np.clip(pre_clip, -5.0, 5.0)
    clipped = (np.abs(pre_clip) > 5.0).mean(axis=0)
    print("force_clipped_fraction", [round(float(x), 6) for x in clipped])
    print("force_norm_mean", [round(float(x), 4) for x in norm_force.mean(axis=0)])
    print("force_norm_std", [round(float(x), 4) for x in norm_force.std(axis=0)])
    print("force_norm_absmax", [round(float(x), 4) for x in np.abs(norm_force).max(axis=0)])
    # The stored proprio is float32; confirm the normalised result is unchanged
    # at that precision, since that is what the dataloader actually feeds.
    pre32 = zeroed.astype(np.float32).astype(np.float64) * scale[8:] + offset[8:]
    print("float32_roundtrip_norm_max_abs_diff",
          float(np.abs(np.clip(pre32, -5, 5) - norm_force).max()))

    out.mkdir(parents=True, exist_ok=True)
    link = out / "frames.dat"
    if not link.exists():
        link.symlink_to(pathlib.Path("..") / "a3_fw" / "frames.dat")
    target = link.resolve()
    expect = n * 224 * 448 * 3
    if target.stat().st_size != expect:
        die("frames.dat symlink resolves to %d B, expected %d B" % (target.stat().st_size, expect))
    print("frames.dat symlink ->", str(target), "bytes", target.stat().st_size)

    payload = {}
    for key in base.files:
        if key == "proprio":
            continue
        payload[key] = np.asarray(base[key])
    payload["proprio"] = proprio14
    np.savez_compressed(out / "arrays.npz", **payload)

    stats_out = {"action": {"default": action_stats}, "state": {"default": state14}}
    (out / "dataset_stats.json").write_text(json.dumps(stats_out, indent=2), encoding="utf-8")

    summary = {
        "version": "v1-force-aware",
        "zero_mode": mode,
        "source_base": str(BASE),
        "source_force": str(FORCE),
        "rows": n,
        "episodes": n_ep,
        "proprio_dim": 14,
        "proprio_layout": ["eef_pos_x", "eef_pos_y", "eef_pos_z",
                           "eef_aa_x", "eef_aa_y", "eef_aa_z",
                           "gripper_finger_a", "gripper_finger_b"] + V6_NAMES,
        "force_channels_in_raw18": FORCE_V6,
        "force_zeroing": ref_desc,
        "force_zero_reference": ref_value,
        "force_normalisation": "residual / residual_std, clip [-5,5]",
        "force_norm_via": "stock min/max with global_min=-std, global_max=+std",
        "residual_std": std.tolist(),
        "residual_mean": mean.tolist(),
        "force_clipped_fraction": clipped.tolist(),
        "per_episode_zero_drift_vs_global_in_episode_std_units": drift.tolist(),
        "frames_dat": "symlink to ../a3_fw/frames.dat",
        "proprio_dtype": "float32",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    back = np.load(out / "arrays.npz", allow_pickle=True)
    if back["proprio"].shape != (n, 14):
        die("readback proprio shape %s" % (back["proprio"].shape,))
    if not np.array_equal(back["proprio"][:, :8], bp.astype(np.float32)):
        die("readback first 8 columns differ from base")
    if not np.array_equal(back["proprio"][:, 8:], zeroed.astype(np.float32)):
        die("readback force columns differ")
    for key in payload:
        if key != "proprio" and not np.array_equal(np.asarray(back[key]), payload[key]):
            die("readback key %s differs" % key)
    rt = json.loads((out / "dataset_stats.json").read_text())
    if len(rt["state"]["default"]["global_min"]) != 14:
        die("readback stats are not 14 wide")
    print("readback ok; keys", sorted(back.files))
    print("arrays.npz bytes", (out / "arrays.npz").stat().st_size)
    print("BUILD_V1_DONE mode=" + mode)


main()