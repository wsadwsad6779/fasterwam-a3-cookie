"""Stage B2/E0: the (model x control) closed-loop matrix.

    model   in {raw, lora}                      -- released Faster-WAM vs merged A3 LoRA
    control in {B0, C0, C1..C5, safety, force, guard, full}
        B0            : bare policy -- interface only, safety layer off. The
                        default condition; always paired with C0.
        C0 / safety   : interface + safety layer only
        C1..C5        : full wrapper, one capability handed back per stage
                        (transport xyz -> approach xyz -> grasp target ->
                         descend/close -> release/recovery); C5 is all of them and
                        must reproduce C0
        force/guard/full : the historical cumulative ablation, kept as a control

Authority moves one capability at a time, never by adding task rules.

B0 and C0 are the same episode with and without the safety layer: same weights,
seed, reset and action stream, so the only difference is what the layer rewrote.
That difference is not small -- r4's C0 seed 22 scored 3/10 with 112 of 600
steps rewritten -- so every run reports both, and the B0 record carries
`safety_would_have_*` counters for the intervention the layer would have made.

Position perturbations use `--batch-offset`, a rigid translation of the whole
batch; per-cookie `--position-noise` destroys the 2.5 mm/0.4 mm clearances (see
`scene_audit.py`) and is stopped by the scene sanity gate.

Every cell uses the same seeds and the same reset, so cells are paired.
One model per process: two 5.56B checkpoints do not belong in one 48 GB card.

Per step we log the adapted model action *before* the control layer, the action
actually executed, and the canonical action the simulator applied, so control
authority can be measured afterwards instead of assumed. Simulator-truth reads
by the controller are counted through `OracleProbe`; the logger itself never
touches `env.data` or `env.cookie_positions`.

    python3 run_matrix.py --model lora --controls B0,C0 --seeds 11 21 22
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

import a3_wrap
import cookie_truth_log
import env_factory
import safety_wrap
from common import describe_policy, load_policy
from oracle_probe import OracleProbe


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["raw", "lora"])
    ap.add_argument("--controls", default="B0,C0",
                    help="B0 = bare policy, interface only, no safety layer "
                         "(the default condition, paired with C0 on the same "
                         "seeds); C0/safety = interface + safety only; C1..C5 = "
                         "one capability released per stage (C5 releases all and "
                         "must match C0); full/guard/force = historical profiles")
    ap.add_argument("--seeds", type=int, nargs="+", default=[11, 21, 22, 23, 24])
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--replan", type=int, default=10)
    ap.add_argument("--position-noise", type=float, default=0.0,
                    help="cookie xy noise in metres; 0 keeps the stage-B fixed scene")
    ap.add_argument("--yaw-noise", type=float, default=0.0)
    ap.add_argument("--batch-offset", type=float, default=0.0,
                    help="rigid xy offset of the whole cookie batch; keeps the "
                         "2.5 mm/0.4 mm clearances intact, unlike per-cookie noise")
    ap.add_argument("--randomize-boxes", action="store_true")
    ap.add_argument("--allow-invalid-scene", action="store_true",
                    help="run even if the reset scene has overlapping or escaped cookies")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--adapter", default=None,
                    help="adapter.pt to apply on top of --ckpt; evaluates a "
                         "light-state run without writing an 11 GB merged file")
    ap.add_argument("--lora-config", default=None,
                    help="resolved config.yaml whose `lora:` block describes the "
                         "adapter structure (default: the A3 task config)")
    ap.add_argument("--stats", default=None)
    ap.add_argument("--instruction", default=None)
    ap.add_argument("--gripper-mode", default=None, choices=["pm1", "libero01"],
                    help="default: per checkpoint (raw=libero01, lora=pm1). "
                         "Stage B1 measures which one is right; this only replays it.")
    ap.add_argument("--soft-force", type=float, default=None)
    ap.add_argument("--hard-force", type=float, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--video", action="store_true")
    return ap.parse_args()


def make_limits(args):
    lim = safety_wrap.SafetyLimits()
    if args.soft_force is not None:
        lim.soft_force_n = args.soft_force
    if args.hard_force is not None:
        lim.hard_force_n = args.hard_force
    return lim


def resolve_control(control):
    """Map a control name to (recorded name, wrapped?, profile, released caps).

    `B0` is the bare policy: interface only (frame adapter + pm1 gripper
    binarisation), no safety layer. It is the default condition from here on, and
    it is paired with `C0` on the same seed so the intervention rate stops being
    invisible.

    `safety` and `C0` are the same condition: interface + safety layer only.
    `C1..C4` use the full profile, keep the safety layer, and hand one
    capability at a time back to the policy. `C5` releases everything, so it is
    routed through the safety path on purpose: C5 must equal C0 action for
    action, and a separate code path would hide a leak. `full/guard/force` stay
    available as the historical ablation.
    """
    name = control.strip()
    upper = name.upper()
    if upper == "B0":
        return "B0", False, a3_wrap.PROFILES["full"], frozenset()
    if name.lower() == "safety" or upper == "C0":
        return "C0", False, a3_wrap.PROFILES["full"], frozenset()
    if upper == "C5":
        # C5 goes *through* the wrapper with every capability released, then the
        # safety layer, so it must reproduce C0. Routing it down the C0 branch
        # would skip the wrapper and make the leak check vacuous.
        return "C5", True, a3_wrap.PROFILES["full"], a3_wrap.RELEASE_STAGES["C5"]
    if upper in a3_wrap.RELEASE_STAGES:
        return upper, True, a3_wrap.PROFILES["full"], a3_wrap.RELEASE_STAGES[upper]
    if name in a3_wrap.PROFILES and name != "baseline":
        return name, True, a3_wrap.PROFILES[name], frozenset()
    raise SystemExit(f"unknown control {control}")


def run_one(policy, args, control, seed, lim, out_dir, src_pts):
    import infer_a3_worker as W

    control_name, wrapped, profile, released = resolve_control(control)
    # Images are the policy input, so cameras are always on; --video only
    # decides whether the front view is written to disk.
    env = env_factory.make_env(position_noise_m=args.position_noise,
                               yaw_noise_rad=args.yaw_noise,
                               render_cameras=True)
    probe = OracleProbe(env)
    W.set_global_seed(int(seed), get_worker_init_fn=False)
    obs, info = env_factory.reset(env, seed, randomize_cookies=True,
                                  randomize_boxes=args.randomize_boxes,
                                  batch_offset_m=args.batch_offset)
    # A scene with overlapping or escaped cookies cannot be scored, so it stops
    # here instead of producing a number that looks like a policy failure.
    sanity = env_factory.scene_sanity(env)
    if not sanity["ok"]:
        msg = (f"scene invalid at noise={args.position_noise} "
               f"offset={args.batch_offset} seed={seed}: {sanity}")
        if not args.allow_invalid_scene:
            env.close()
            raise SystemExit(msg)
        print(f"[scene] WARNING {msg}", flush=True)
    # `info` is rebound by every step; keep the reset-time offset for the record.
    applied_offset = [float(v) for v in info.get("batch_offset_xy_m", (0.0, 0.0))]
    # The policy object is reused across cells, so its call counter has to be
    # reset per episode or `replans` and `chunk_idx` inherit the previous cell.
    policy.calls = 0
    cookie_xyz0 = cookie_truth_log.initial(env)

    ctl_state = a3_wrap.new_state() if wrapped else safety_wrap.new_state()
    # C1..C4 keep the safety layer while the wrapper hands over capabilities;
    # `safety_st` is unused on the C0/C5 path, which runs safety_wrap itself.
    staged = wrapped and control_name.startswith("C")
    safety_st = safety_wrap.new_state() if staged else None
    # B0 executes nothing but the interface output, but the safety layer is
    # still run on a throwaway copy so the intervention it *would* have made is
    # counted instead of silently absent.
    bare = control_name == "B0"
    shadow_st = safety_wrap.new_state() if bare else None
    shadow_changed = []
    nan_abort = -1

    pending, chunk_idx, pre_log, post_log, phase_log = [], [], [], [], []
    opening, applied, force_in, force_out, dist, eef = [], [], [], [], [], []
    applied_canonical, post_wrap_log, cookie_log = [], [], []
    wrist = []
    oracle_cookie, oracle_data, oracle_sensor, frames = [], [], [], []

    probe.reset_counts()
    t0 = time.time()
    step = 0
    for step in range(args.max_steps):
        if not pending:
            chunk7 = policy.chunk(obs, seed)
            pending = policy.adapted(chunk7, obs)[:args.replan]
            chunk_idx.extend([policy.calls - 1] * len(pending))

        pre = np.asarray(pending.pop(0), dtype=np.float64)
        # post_w keeps the wrapper output before the safety layer, so the two
        # intervention rates are attributable. All three arrays are 14-d in the
        # same units and at the same timestep.
        if wrapped:
            post_w = a3_wrap.apply(pre.copy(), obs, probe, ctl_state, profile, released)
            post = safety_wrap.apply(post_w, obs, safety_st, lim) if staged else post_w
            phase_log.append(a3_wrap.PHASES.index(ctl_state["phase"]))
        else:
            post_w = pre.copy()
            if bare:
                post = pre.copy()
                shadow_changed.append(bool(np.any(
                    safety_wrap.apply(pre.copy(), obs, shadow_st, lim) != pre)))
            else:
                post = safety_wrap.apply(pre, obs, ctl_state, lim)
            phase_log.append(-1)

        obs, _, terminated, truncated, info = env.step(post)
        tcp = np.asarray(obs["observation.eef_pose"][:3], dtype=np.float64)
        if not np.all(np.isfinite(tcp)):
            # Nothing is being clamped, so an unstable state has to end the
            # episode and be reported instead of quietly poisoning the log.
            nan_abort = step
            print(f"[{policy.name}/{control}/{seed}] non-finite eef at "
                  f"step={step + 1}, aborting episode", flush=True)
            break

        pre_log.append(pre)
        post_log.append(np.asarray(post, dtype=np.float64))
        post_wrap_log.append(np.asarray(post_w, dtype=np.float64))
        opening.append(float(obs["observation.state"][7]))
        applied.append(float(info["applied_action"][7]))
        applied_canonical.append(np.asarray(info["applied_action"], dtype=np.float64).copy())
        f = np.asarray(obs[safety_wrap.FORCE_KEY], dtype=np.float64)
        force_in.append(float(f[safety_wrap.L_INNER]))
        force_out.append(float(f[safety_wrap.L_OUTER]))
        wrist.append(np.asarray(f[0:3], dtype=np.float64))
        dist.append(float(np.min(np.linalg.norm(src_pts - tcp[:2][None, :], axis=1))))
        eef.append(tcp)
        # Diagnostics only: same timestep as tcp, never fed back to any controller.
        cookie_log.append(cookie_truth_log.snapshot(env, step))
        oracle_cookie.append(probe.counts["cookie_positions"])
        oracle_data.append(probe.counts["data"])
        oracle_sensor.append(probe.counts["_sensor"])
        if args.video:
            frames.append(np.asarray(obs["observation.images.front"]).copy())

        if (step + 1) % 50 == 0:
            print(f"[{policy.name}/{control}/{seed}] step={step + 1} "
                  f"in={info.get('cookies_in_target')} xyz={np.round(tcp, 3).tolist()} "
                  f"dist={dist[-1]:.3f} f=({force_in[-1]:.2f},{force_out[-1]:.2f}) "
                  f"oracle=({oracle_cookie[-1]},{oracle_data[-1]})", flush=True)
        if terminated or truncated:
            break

    duration = time.time() - t0
    pre_arr = np.asarray(pre_log)
    post_arr = np.asarray(post_log)
    changed = np.any(np.abs(pre_arr - post_arr) > 1e-9, axis=1)

    np.savez_compressed(
        out_dir / f"{policy.name}_{control_name}_{seed}.npz",
        pre_14d=pre_arr,
        post_14d=post_arr,
        changed=changed,
        post_wrap_14d=np.asarray(post_wrap_log),
        cookie_xyz=np.asarray([c for _, c in cookie_log]),
        cookie_step=np.asarray([s for s, _ in cookie_log]),
        cookie_xyz0=cookie_xyz0,
        chunk_idx=np.asarray(chunk_idx[:len(post_arr)]),
        phase=np.asarray(phase_log, dtype=np.int8),
        opening=np.asarray(opening),
        applied_opening=np.asarray(applied),
        applied_canonical=np.asarray(applied_canonical),
        pad_force_inner=np.asarray(force_in),
        pad_force_outer=np.asarray(force_out),
        dist_to_source=np.asarray(dist),
        eef_xyz=np.asarray(eef),
        wrist_force=np.asarray(wrist),
        oracle_cookie=np.asarray(oracle_cookie),
        oracle_data=np.asarray(oracle_data),
        oracle_sensor=np.asarray(oracle_sensor),
    )
    if args.video:
        import imageio

        imageio.mimsave(out_dir / f"{policy.name}_{control_name}_{seed}_front.mp4", frames, fps=30)

    record = {
        "model": policy.name,
        "control": control_name,
        "wrapped": bool(wrapped),
        "released": sorted(released),
        "safety_layer": bool((staged or not wrapped) and not bare),
        "seed": int(seed),
        "steps": step + 1,
        "replans": policy.calls,
        "success": bool(info.get("success", False)),
        "cookies_in_target": int(info.get("cookies_in_target", -1)),
        "final_phase": ctl_state.get("phase", "n/a") if wrapped else "n/a",
        "max_pad_force": round(float(max(max(force_in), max(force_out))), 3),
        "max_wrist_force": round(float(max(np.linalg.norm(np.asarray(wrist), axis=1))), 2),
        "verbatim_pct": round(float(100.0 * (~changed).mean()), 2),
        "oracle_cookie_reads": int(oracle_cookie[-1]),
        "oracle_data_reads": int(oracle_data[-1]),
        "sensor_reads": int(oracle_sensor[-1]),
        "estop_steps": int(ctl_state.get("hard", 0)),
        "binfloor_steps": int(ctl_state.get("binfloor", 0)),
        # B0 only: what the safety layer would have rewritten had it been in the
        # loop. Counted on a throwaway copy, never executed.
        "safety_would_have_changed_pct": round(100.0 * float(np.mean(shadow_changed)), 2)
                                        if bare and shadow_changed else None,
        "safety_would_have_hard": int(shadow_st.get("hard", 0)) if bare else None,
        "safety_would_have_binfloor": int(shadow_st.get("binfloor", 0)) if bare else None,
        "nan_abort_step": int(nan_abort),
        "duration_s": round(duration, 1),
        "position_noise_m": args.position_noise,
        "yaw_noise_rad": args.yaw_noise,
        "batch_offset_m": args.batch_offset,
        "batch_offset_xy_m": applied_offset,
        "randomize_boxes": bool(args.randomize_boxes),
        "scene_ok": bool(sanity["ok"]),
        "scene_max_penetration_mm": sanity["max_penetration_mm"],
        "scene_cookies_outside_bin": sanity["cookies_outside_bin"],
        "gripper_mode": policy.gripper_mode,
    }
    env.close()
    return record


def main():
    args = parse_args()
    if args.model == "lora" and not args.ckpt:
        raise SystemExit("refusing to run lora without an explicit --ckpt")
    if args.adapter and not args.ckpt:
        raise SystemExit("--adapter needs --ckpt: the frozen base it was trained on")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = describe_policy(args.model, args.ckpt, args.stats, args.adapter)
    policy = load_policy(args.model, args.ckpt, args.stats, args.instruction,
                         mode=args.gripper_mode, adapter=args.adapter,
                         lora_config=args.lora_config)
    manifest.update({"replan": args.replan, "max_steps": args.max_steps,
                     "controls": args.controls, "seeds": args.seeds,
                     "gripper_mode": policy.gripper_mode})
    if args.adapter:
        manifest.update({"adapter_path": policy.adapter_path,
                         "adapter_meta": policy.adapter_meta,
                         "lora": policy.lora})
    (out_dir / f"manifest_{args.model}.json").write_text(json.dumps(manifest, indent=2))
    print(f"[matrix] manifest={json.dumps(manifest['normalizer'])}", flush=True)

    ck = manifest["checkpoint"]
    print("[matrix] loaded_ckpt=%s size=%s sha256_head=%s"
          % (ck["path"], ck["size"], ck["sha256_head"]), flush=True)

    lim = make_limits(args)
    print(f"[matrix] limits soft={lim.soft_force_n} hard={lim.hard_force_n}", flush=True)

    from a3_dual_arm_sim.config import load_config

    src_pts = np.asarray(
        load_config(env_factory.DEFAULT_CONFIG).cookie_transfer.cookie_source_positions_m,
        dtype=np.float64,
    )

    records = []
    for control in [c.strip() for c in args.controls.split(",") if c.strip()]:
        resolve_control(control)  # fail before the first episode, not after it
        for seed in args.seeds:
            rec = run_one(policy, args, control, seed, lim, out_dir, src_pts)
            records.append(rec)
            print(f"[matrix] {json.dumps(rec)}", flush=True)
            (out_dir / f"summary_{args.model}.json").write_text(json.dumps(records, indent=2))

    print(f"[matrix] done {len(records)} runs -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
