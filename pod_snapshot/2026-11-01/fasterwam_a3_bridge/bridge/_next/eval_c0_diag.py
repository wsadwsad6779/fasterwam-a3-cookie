"""C0 diagnostic harness: same execution semantics, more evidence.

This wraps `run_matrix.run_one` without changing what the simulator executes:
the policy still runs `infer_a3_worker.predict_action_chunk`, the adapter still
consumes the binarized gripper, and the safety layer still sees the same
14-D action. What changes is only *observation*:

  * the instruction is forced to the training sentence and asserted per chunk,
  * the denormalized action is captured *before* the pm1 binarization, so the
    continuous gripper signal and its distance to the 0 threshold survive,
  * every chunk writes all 32 predicted steps (raw and binarized, pre- and
    post-adapter), plus the eef quaternion used for the rotation conversion,
  * scene seed and policy seed are separated: `--scene-seed` fixes the initial
    scene while `--seeds` only drives the policy's own RNG,
  * an explicit rigid `--batch-offset-xy` freezes a shifted scene,
  * `--repeats` runs the same seed twice and writes a determinism report.

The pm1/full/safety path is untouched: `raw` is a copy taken before the
in-place rewrite, so no diagnostics can leak back into the executed action.

    python3 eval_c0_diag.py --model lora --ckpt ... --stats ... \
        --gripper-mode pm1 --instruction 'transfer 10 cookies into target box' \
        --controls C0 --seeds 11 21 22 --replan 10 --max-steps 600 \
        --scene-seed 11 --out /workspace/a3_eval/r3_c0_diag
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

TRAIN_INSTRUCTION = "transfer 10 cookies into target box"

# Must be set before infer_a3_worker is imported anywhere (common/run_matrix
# import it lazily inside functions, but this is the only safe ordering).
os.environ["FASTERWAM_A3_INSTRUCTION"] = TRAIN_INSTRUCTION

import common  # noqa: E402
import env_factory  # noqa: E402
import run_matrix as RM  # noqa: E402

DIAG = {"policy": None, "chunks": None, "raw": None, "step": 0, "rep": 0}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["raw", "lora"])
    ap.add_argument("--controls", default="C0")
    ap.add_argument("--seeds", type=int, nargs="+", default=[11, 21, 22],
                    help="policy seeds: drive set_global_seed and the model RNG")
    ap.add_argument("--scene-seed", type=int, default=None,
                    help="fixes the initial scene; defaults to the policy seed")
    ap.add_argument("--repeats", type=int, default=1,
                    help="run each (control, seed) this many times and diff runs")
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--replan", type=int, default=10)
    ap.add_argument("--position-noise", type=float, default=0.0)
    ap.add_argument("--yaw-noise", type=float, default=0.0)
    ap.add_argument("--batch-offset", type=float, default=0.0)
    ap.add_argument("--batch-offset-xy", type=float, nargs=2, default=None,
                    help="explicit rigid xy offset in metres; freezes a shifted scene")
    ap.add_argument("--randomize-boxes", action="store_true")
    ap.add_argument("--allow-invalid-scene", action="store_true")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--stats", default=None)
    ap.add_argument("--instruction", default=TRAIN_INSTRUCTION)
    ap.add_argument("--gripper-mode", default=None, choices=["pm1", "libero01"])
    ap.add_argument("--soft-force", type=float, default=None)
    ap.add_argument("--hard-force", type=float, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--video", action="store_true")
    args = ap.parse_args()
    if args.instruction != TRAIN_INSTRUCTION:
        raise SystemExit(f"--instruction must be the training sentence: {TRAIN_INSTRUCTION!r}")
    if args.batch_offset_xy is not None and args.batch_offset:
        raise SystemExit("use either --batch-offset or --batch-offset-xy, not both")
    return args


def sha256_file(path, nbytes=None):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        remaining = nbytes
        while True:
            chunk = f.read(1 << 22 if remaining is None else min(1 << 22, remaining))
            if not chunk:
                break
            h.update(chunk)
            if remaining is not None:
                remaining -= len(chunk)
                if remaining <= 0:
                    break
    return h.hexdigest()


def install_hooks(policy, scene_seed, offset_xy):
    import infer_a3_worker as W

    orig_make_env = env_factory.make_env

    def make_env(*a, **k):
        env = orig_make_env(*a, **k)
        orig_step = env.step

        def step(action):
            out = orig_step(action)
            DIAG["step"] += 1
            return out

        env.step = step
        return env

    orig_reset = env_factory.reset

    def reset(env, seed, randomize_cookies=True, randomize_boxes=False,
              batch_offset_m=0.0):
        scene = int(scene_seed) if scene_seed is not None else int(seed)
        obs, info = orig_reset(env, scene, randomize_cookies=randomize_cookies,
                               randomize_boxes=randomize_boxes,
                               batch_offset_m=batch_offset_m)
        if offset_xy is not None and np.any(offset_xy):
            off = env_factory.apply_batch_offset(env, 0.0,
                                                 offset=np.asarray(offset_xy, dtype=np.float64))
            obs = env._observation()
            info["batch_offset_xy_m"] = off.tolist()
        # env.reset may itself step the simulator; the episode clock starts now.
        DIAG["step"] = 0
        return obs, info

    orig_denorm = W.denormalize_action

    def denorm(action, processor):
        out = orig_denorm(action, processor)
        DIAG["raw"] = np.array(out, copy=True)
        return out

    orig_pred = W.predict_action_chunk

    def predict(obs, model, processor, cfg, seed, model_device):
        instruction = W.INSTRUCTION
        if instruction != TRAIN_INSTRUCTION:
            raise RuntimeError(f"instruction drift: {instruction!r}")
        quat = np.asarray(obs["observation.eef_pose"][3:7], dtype=np.float64).copy()
        DIAG["raw"] = None
        rgb, bin_chunk = orig_pred(obs, model, processor, cfg, seed, model_device)
        raw_chunk = DIAG["raw"]
        if raw_chunk is None:
            raise RuntimeError("denormalize_action hook did not fire")
        raw = np.asarray(raw_chunk[0], dtype=np.float64)
        binned = np.asarray(bin_chunk, dtype=np.float64)
        rot = W.quat_wxyz_to_matrix(quat)
        adapted_raw = np.stack([policy.adapter.convert(a, rot) for a in raw])
        adapted_bin = np.stack([policy.adapter.convert(a, rot) for a in binned])
        DIAG["chunks"].append({
            "step": int(DIAG["step"]),
            "instruction": instruction,
            "quat": quat,
            "raw": raw,
            "bin": binned,
            "adapted_raw": adapted_raw,
            "adapted_bin": adapted_bin,
            "rgb_sha": hashlib.sha256(np.ascontiguousarray(rgb)).hexdigest()[:16],
        })
        return rgb, bin_chunk

    env_factory.make_env = make_env
    env_factory.reset = reset
    W.denormalize_action = denorm
    W.predict_action_chunk = predict
    return {
        "env_factory.make_env": orig_make_env,
        "env_factory.reset": orig_reset,
        "W.denormalize_action": orig_denorm,
        "W.predict_action_chunk": orig_pred,
    }


def save_diag(path, chunks, policy_name, control, seed, rep):
    if not chunks:
        return None
    np.savez_compressed(
        path,
        chunk_step=np.asarray([c["step"] for c in chunks], dtype=np.int64),
        raw=np.stack([c["raw"] for c in chunks]),
        bin=np.stack([c["bin"] for c in chunks]),
        adapted_raw=np.stack([c["adapted_raw"] for c in chunks]),
        adapted_bin=np.stack([c["adapted_bin"] for c in chunks]),
        chunk_quat=np.stack([c["quat"] for c in chunks]),
        chunk_instruction=np.asarray([c["instruction"] for c in chunks]),
        chunk_rgb_sha=np.asarray([c["rgb_sha"] for c in chunks]),
        raw_gripper=np.stack([c["raw"][:, 6] for c in chunks]),
        bin_gripper=np.stack([c["bin"][:, 6] for c in chunks]),
        meta=json.dumps({"model": policy_name, "control": control,
                         "policy_seed": int(seed), "rep": int(rep),
                         "instruction": TRAIN_INSTRUCTION}),
    )
    return path


def compare_runs(rep_dirs, policy_name, control, seed):
    """Diff two repeats of the same seed: predicted chunks and executed states."""
    out = {"policy_seed": int(seed), "control": control}
    diags, mains = [], []
    for d in rep_dirs:
        diag = d / f"diag_{policy_name}_{control}_{seed}_r{d.name[-1]}.npz"
        main = d / f"{policy_name}_{control}_{seed}.npz"
        diags.append(diag if diag.exists() else None)
        mains.append(main if main.exists() else None)
    if None in diags or None in mains:
        out["error"] = "missing artifacts"
        return out

    def diff_npz(a, b, keys):
        res = {}
        for k in keys:
            if k not in a or k not in b:
                continue
            x, y = a[k], b[k]
            if x.shape != y.shape:
                res[k] = {"shape_mismatch": [list(x.shape), list(y.shape)]}
                continue
            if x.dtype.kind in "USO":
                res[k] = {"equal": bool(np.array_equal(x, y))}
            else:
                res[k] = {"equal": bool(np.array_equal(x, y)),
                          "max_abs_diff": float(np.max(np.abs(x - y))) if x.size else 0.0}
        return res

    with np.load(diags[0], allow_pickle=False) as a, np.load(diags[1], allow_pickle=False) as b:
        out["predicted"] = diff_npz(a, b, ["raw", "bin", "adapted_raw", "adapted_bin",
                                          "chunk_quat", "chunk_step", "raw_gripper"])
    with np.load(mains[0], allow_pickle=False) as a, np.load(mains[1], allow_pickle=False) as b:
        out["executed"] = diff_npz(a, b, ["post_14d", "opening", "applied_canonical",
                                          "eef_xyz", "cookie_xyz", "pad_force_inner"])
    out["reproduced"] = bool(
        all(v.get("equal", False) for v in out["predicted"].values())
        and all(v.get("equal", False) for v in out["executed"].values()))
    return out


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = common.describe_policy(args.model, args.ckpt, args.stats)
    manifest.update({
        "instruction_expected": TRAIN_INSTRUCTION,
        "instruction_arg": args.instruction,
        "stats_sha256": sha256_file(Path(args.stats) if args.stats else common.MODELS[args.model]["stats"]),
        "ckpt_sha256_head": manifest["checkpoint"]["sha256_head"],
        "replan": args.replan,
        "max_steps": args.max_steps,
        "policy_seeds": args.seeds,
        "scene_seed": args.scene_seed,
        "repeats": args.repeats,
        "batch_offset_m": args.batch_offset,
        "batch_offset_xy_m": args.batch_offset_xy,
        "position_noise_m": args.position_noise,
        "yaw_noise_rad": args.yaw_noise,
        "controls": args.controls,
    })
    code_files = ["eval_c0_diag.py", "run_matrix.py", "common.py", "env_factory.py",
                  "safety_wrap.py", "a3_wrap.py", "infer_a3_worker.py",
                  "action_adapter_v3.py", "cookie_truth_log.py"]
    here = Path(__file__).resolve().parent
    code = {}
    for name in code_files:
        for d in (here, here.parent):
            p = d / name
            if p.exists():
                code[name] = sha256_file(p)
                break
    manifest["code_sha256"] = code
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[diag] instruction={TRAIN_INSTRUCTION!r} stats_sha={manifest['stats_sha256'][:16]}",
          flush=True)

    policy = common.load_policy(args.model, args.ckpt, args.stats, args.instruction,
                                mode=args.gripper_mode)
    DIAG["policy"] = policy
    install_hooks(policy, args.scene_seed, args.batch_offset_xy)

    lim = RM.make_limits(args)
    from a3_dual_arm_sim.config import load_config
    src_pts = np.asarray(
        load_config(env_factory.DEFAULT_CONFIG).cookie_transfer.cookie_source_positions_m,
        dtype=np.float64,
    )

    records = []
    determinism = []
    rep_dirs = [out_dir / f"rep{r}" for r in range(args.repeats)]
    for d in rep_dirs:
        d.mkdir(exist_ok=True)

    for control in [c.strip() for c in args.controls.split(",") if c.strip()]:
        RM.resolve_control(control)
        for seed in args.seeds:
            for rep, sub in enumerate(rep_dirs):
                DIAG["rep"] = rep
                DIAG["chunks"] = []
                rec = RM.run_one(policy, args, control, seed, lim, sub, src_pts)
                rec["rep"] = rep
                rec["scene_seed"] = int(args.scene_seed) if args.scene_seed is not None else int(seed)
                rec["policy_seed"] = int(seed)
                rec["instruction"] = TRAIN_INSTRUCTION
                save_diag(sub / f"diag_{policy.name}_{control}_{seed}_r{rep}.npz",
                          DIAG["chunks"], policy.name, control, seed, rep)
                records.append(rec)
                print(f"[diag] {json.dumps(rec)}", flush=True)
            if len(rep_dirs) > 1:
                determinism.append(compare_runs(rep_dirs, policy.name, control, seed))
                print(f"[diag] determinism {json.dumps(determinism[-1])}", flush=True)

    (out_dir / "summary.json").write_text(json.dumps(records, indent=2))
    if determinism:
        (out_dir / "determinism.json").write_text(json.dumps(determinism, indent=2))
    manifest["finished"] = True
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[diag] done {len(records)} runs -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
