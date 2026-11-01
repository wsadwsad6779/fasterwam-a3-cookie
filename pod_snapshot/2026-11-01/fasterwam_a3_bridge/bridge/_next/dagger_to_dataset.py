"""Convert DAgger takeover rollouts into the A3 training layout.

Mirrors `finetune/prepare_a3_arrays.py` exactly: same action formula (world
delta / 0.05, rotation delta / 0.5, gripper from the measured opening with the
1e-4 deadband), same 8-dim proprio, same `frames.dat` layout (uint8,
(N, 224, 448, 3), front | left_wrist, 224 bilinear). The corrective episodes
therefore drop into the existing trainer with no code change.

Labels are recomputed from the replayed poses rather than copied from the
expert's 14-dim command, because that is what the training arrays contain. The
expert command is kept next to them and the two are compared, so a mismatch in
action representation shows up as a number instead of silently poisoning the
dataset.

    python3 dagger_to_dataset.py --takes 22:40 21:200 --out /workspace/a3_data/a3_fw_dagger
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

IMG = 224
POS_SCALE, ROT_SCALE = 0.05, 0.5
GRIP_EPS = 1e-4
FRAME_BYTES = IMG * IMG * 2 * 3


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", nargs="+", default=["/workspace/fasterwam_a3_bridge/bridge/outputs/dagger"])
    ap.add_argument("--takes", nargs="+", required=True, help="seed:take ...")
    ap.add_argument("--out", required=True)
    return ap.parse_args()


def quat_wxyz_to_aa(q):
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    x, y, z, w = q[..., 1], q[..., 2], q[..., 3], q[..., 0]
    w = np.clip(w, -1.0, 1.0)
    den = np.sqrt(np.maximum(0.0, 1.0 - w * w))
    ang = 2.0 * np.arccos(w)
    out = np.zeros(q.shape[:-1] + (3,))
    ok = den > 1e-8
    out[ok] = np.stack([x, y, z], -1)[ok] * (ang[ok] / den[ok])[:, None]
    return out


def build_episode(take_dir):
    rec = np.load(take_dir / "recover.npz")
    eef = np.asarray(rec["eef_pose"], dtype=np.float64)
    cmd = np.asarray(rec["action14"], dtype=np.float64)
    opening = np.asarray(rec["opening"], dtype=np.float64)
    applied = np.asarray(rec["applied_joint"], dtype=np.float64)
    frames_path = take_dir / "frames.dat"
    if not frames_path.exists():
        raise SystemExit(f"{take_dir}: frames.dat missing (rerun with --dump-frames)")

    n_states, n_actions = len(eef), len(cmd)
    if n_states != n_actions + 1:
        raise SystemExit(f"{take_dir}: expected states=actions+1, got {n_states}/{n_actions}")
    if len(opening) != n_states or len(applied) != n_actions:
        raise SystemExit(f"{take_dir}: row mismatch opening={len(opening)} applied={len(applied)}")
    if n_actions < 2:
        raise SystemExit(f"{take_dir}: too short ({n_actions} actions)")

    T = n_actions
    p, quat = eef[:T, :3], eef[:T, 3:7]
    R = Rotation.from_quat(np.stack([quat[:, 1], quat[:, 2], quat[:, 3], quat[:, 0]], -1)).as_matrix()
    dR = R[1:] @ np.transpose(R[:-1], (0, 2, 1))
    pos = np.diff(p, axis=0) / POS_SCALE
    rot = Rotation.from_matrix(dR).as_rotvec() / ROT_SCALE

    op = opening[:T]
    d = np.diff(opening)
    g = np.full(T, -1.0, dtype=np.float32)
    for t in range(T - 1):
        if d[t] < -GRIP_EPS:
            g[t] = 1.0
        elif d[t] > GRIP_EPS:
            g[t] = -1.0
        else:
            g[t] = g[t - 1] if t > 0 else -1.0
    g[T - 1] = g[T - 2]

    act = np.zeros((T, 7), dtype=np.float32)
    act[:T - 1, :3] = pos
    act[:T - 1, 3:6] = rot
    act[:T - 1, 6] = g[:T - 1]
    act[T - 1] = act[T - 2]

    prop = np.zeros((T, 8), dtype=np.float32)
    prop[:, :3] = p
    prop[:, 3:6] = quat_wxyz_to_aa(quat)
    prop[:, 6] = 0.04 * op
    prop[:, 7] = -0.04 * op

    frames = np.memmap(frames_path, dtype=np.uint8, mode="r",
                       shape=(max(frames_path.stat().st_size // FRAME_BYTES, T),
                              IMG, IMG * 2, 3))[:T]

    # The 14-dim expert command is in the env's own units (cartesian scale,
    # absolute gripper target in [-1, 1]), while the label is in the dataset's
    # units (world delta / 0.05, opening direction). Compare direction and
    # scale ratio instead of assuming identical units.
    ref_pos, ref_rot = cmd[:T - 1, :3], cmd[:T - 1, 3:6]
    lab_pos, lab_rot = act[:T - 1, :3], act[:T - 1, 3:6]

    def _cos_mean(a, b):
        na, nb = np.linalg.norm(a, axis=-1), np.linalg.norm(b, axis=-1)
        ok = (na > 1e-6) & (nb > 1e-6)
        if not ok.any():
            return None
        return round(float(np.mean(np.sum(a[ok] * b[ok], -1) / (na[ok] * nb[ok]))), 3)

    # Expert gripper channel is an absolute target in [-1, 1]; a falling target
    # means closing, which the label encodes as +1.
    dcmd = np.diff(cmd[:, 6])
    grip_ref = np.where(dcmd < -1e-3, 1.0, np.where(dcmd > 1e-3, -1.0, 0.0))
    moving = grip_ref != 0
    grip_lab = act[:T - 1, 6]
    check = {
        "pos_scale_ratio": round(float(np.abs(lab_pos).mean() / max(np.abs(ref_pos).mean(), 1e-9)), 3),
        "pos_cos": _cos_mean(lab_pos, ref_pos),
        "rot_scale_ratio": round(float(np.abs(lab_rot).mean() / max(np.abs(ref_rot).mean(), 1e-9)), 3),
        "rot_cos": _cos_mean(lab_rot, ref_rot),
        "grip_dir_agree": round(float((np.sign(grip_lab[moving]) == grip_ref[moving]).mean()), 3) if moving.any() else None,
        "grip_dir_steps": int(moving.sum()),
        "grip_plus_one_frac": round(float((act[:, 6] > 0).mean()), 3),
        "opening_range": [round(float(opening[:T].min()), 3), round(float(opening[:T].max()), 3)],
        "saturated_pct": round(100.0 * float((np.abs(act[:T - 1, :6]) >= 0.999).mean()), 2),
        "label_pos_mm_mean": round(float(np.abs(lab_pos).mean() * POS_SCALE * 1000), 3),
        "expert_pos_cmd_mean": round(float(np.abs(ref_pos).mean()), 4),
        "label_rot_mrad_mean": round(float(np.abs(lab_rot).mean() * ROT_SCALE * 1000), 2),
        "expert_rot_cmd_mean": round(float(np.abs(ref_rot).mean()), 4),
    }
    return {
        "action": act, "proprio": prop, "opening": op.astype(np.float32),
        "joint_cmd": applied[:T, 7].astype(np.float32), "frames": frames,
        "check": check, "phases": rec["phase"].tolist(),
        "cookies_in_target_end": int(rec["cookies_in_target"][-1]) if len(rec["cookies_in_target"]) else -1,
    }


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    eps, metas = [], []
    total = 0
    for spec in args.takes:
        seed, take = (int(v) for v in spec.split(":"))
        td = [Path(r) / f"seed{seed}_s{take}" for r in args.root]
        take_dir = next((x for x in td if x.exists()), None)
        if take_dir is None:
            raise SystemExit(f"seed{seed}_s{take}: not found under {args.root}")
        ep = build_episode(take_dir)
        eps.append(ep)
        metas.append({"episode": len(metas), "seed": seed, "take_step": take,
                      "source": str(take_dir), "check": ep["check"],
                      "length": int(len(ep["action"])),
                      "cookies_in_target_end": ep["cookies_in_target_end"],
                      "phase_counts": {p: ep["phases"].count(p) for p in sorted(set(ep["phases"]))}})
        total += len(ep["action"])

    frames = np.memmap(out / "frames.dat", dtype=np.uint8, mode="w+",
                       shape=(total, IMG, IMG * 2, 3))
    action = np.zeros((total, 7), dtype=np.float32)
    prop = np.zeros((total, 8), dtype=np.float32)
    opening = np.zeros(total, dtype=np.float32)
    joint_cmd = np.zeros(total, dtype=np.float32)
    ep_index = np.zeros(total, dtype=np.int64)
    starts, lengths = [], []
    cursor = 0
    for k, ep in enumerate(eps):
        n = len(ep["action"])
        frames[cursor:cursor + n] = ep["frames"]
        action[cursor:cursor + n] = ep["action"]
        prop[cursor:cursor + n] = ep["proprio"]
        opening[cursor:cursor + n] = ep["opening"]
        joint_cmd[cursor:cursor + n] = ep["joint_cmd"]
        ep_index[cursor:cursor + n] = k
        starts.append(cursor)
        lengths.append(n)
        cursor += n
    frames.flush()

    np.savez_compressed(out / "arrays.npz", action=action, proprio=prop, opening=opening,
                        joint_cmd=joint_cmd, episode_index=ep_index,
                        ep_start=np.asarray(starts, dtype=np.int64),
                        ep_len=np.asarray(lengths, dtype=np.int64),
                        ep_ids=np.arange(len(eps), dtype=np.int64))

    base_stats = Path("/workspace/a3_data/a3_fw/dataset_stats.json")
    if base_stats.exists():
        (out / "dataset_stats.json").write_text(base_stats.read_text())

    summary = {
        "episodes": len(eps), "transitions": int(total), "frames": [int(total), IMG, IMG * 2, 3],
        "source_takes": args.takes, "takes": metas,
        "action_mean_abs": np.round(np.abs(action).mean(0), 4).tolist(),
        "gripper_plus_one_frac": round(float((action[:, 6] > 0).mean()), 3),
        "proprio_range": [prop.min(0).round(4).tolist(), prop.max(0).round(4).tolist()],
        "stats_source": str(base_stats) if base_stats.exists() else None,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
