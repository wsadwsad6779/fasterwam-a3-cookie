"""Stage B0: does the gripper command mean what the checkpoint thinks it means?

The per-checkpoint `gripper_mode` (raw=libero01, lora=pm1) was chosen from the
action encoding and the training stats, not from a physical measurement. A sign
error there fails silently: the policy looks decisive, the jaws just never close
on the cookies. So B0 splits the question in three, cheapest first:

  A. servo      -- with no model at all, a3[6]=+1 must OPEN and -1 must CLOSE.
                   Pure env, runs on any machine without weights.
  B. adapter    -- LIBERO +1 (close) must come out of action_adapter_v3 as
                   a3[6]=-1, i.e. the sign flip composes with A.
  C. checkpoint -- optional, needs weights: run one chunk on the reset frame and
                   report what the model actually emits, then step it and check
                   the jaws moved the way the sign claims. A model that asks to
                   close while the gripper starts wide open over an untouched
                   pile is flagged for a mode flip, since approach wants open.

    python3 b0_probe.py --out outputs/b0            # A + B, no weights
    python3 b0_probe.py --model lora --out outputs/b0   # A + B + C

Exit code is 0 only when A and B pass; C is reported, not enforced, because
"which mode is right" is a measurement about the model, not a property of the
bridge.
"""
import argparse
import json
from pathlib import Path

import numpy as np

import env_factory

LEFT_GRIP = 6
STEPS = 120
# make_env binds its default at definition time, so the config travels explicitly.
CFG = env_factory.DEFAULT_CONFIG


def opening_of(obs) -> float:
    return float(obs["observation.state"][7])


def run_servo(seed=11, steps=STEPS):
    """Command the jaw wide open, then fully shut, and measure both."""
    out = {}
    for label, cmd in (("open", 1.0), ("close", -1.0)):
        env = env_factory.make_env(config_path=CFG, render_cameras=False)
        try:
            obs, _ = env_factory.reset(env, seed, randomize_cookies=True)
            start = opening_of(obs)
            a = np.zeros(14, dtype=np.float64)
            a[LEFT_GRIP] = cmd
            for _ in range(steps):
                obs, _, term, trunc, _ = env.step(a)
                if term or trunc:
                    break
            out[label] = {
                "command": cmd,
                "opening_start": round(start, 4),
                "opening_end": round(opening_of(obs), 4),
                "pad_force": round(float(max(obs["observation.force"][12],
                                             obs["observation.force"][13])), 4),
            }
        finally:
            env.close()
    # +1 must end wide, -1 must end shut; the gap is what a real close looks like.
    out["ok"] = (out["open"]["opening_end"] - out["close"]["opening_end"]) > 0.5
    return out


def run_adapter():
    from action_adapter_v3 import FasterWAMA3ActionAdapterV3

    ad = FasterWAMA3ActionAdapterV3()
    rot = np.eye(3)
    cases = {}
    for label, g in (("libero_close", 1.0), ("libero_open", -1.0)):
        a3 = ad.convert(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, g]), rot)
        cases[label] = {
            "libero_gripper": g,
            "a3_grip_dim6": round(float(a3[LEFT_GRIP]), 4),
            "a3_right_dim13": round(float(a3[13]), 4),
        }
    # LIBERO +1 close -> A3 -1 close (the A3 convention is +1 open).
    cases["ok"] = (cases["libero_close"]["a3_grip_dim6"] < 0.0
                   <= cases["libero_open"]["a3_grip_dim6"])
    return cases


def run_checkpoint(model, seed=11, steps=40):
    from common import describe_policy, load_policy

    policy = load_policy(model)
    env = env_factory.make_env(config_path=CFG, render_cameras=True)
    try:
        obs, _ = env_factory.reset(env, seed, randomize_cookies=True)
        start = opening_of(obs)
        chunk7 = policy.chunk(obs, seed)
        adapted = np.asarray(policy.adapted(chunk7, obs), dtype=np.float64)
        grips = adapted[:, LEFT_GRIP]

        trace = []
        for a in adapted[:steps]:
            obs, _, term, trunc, _ = env.step(a)
            trace.append(opening_of(obs))
            if term or trunc:
                break

        return {
            "checkpoint": describe_policy(model)["checkpoint"],
            "gripper_mode": policy.gripper_mode,
            "raw_chunk_grip": {
                "min": round(float(np.min(chunk7[:, 6])), 4),
                "max": round(float(np.max(chunk7[:, 6])), 4),
            },
            "adapted_grip_minus1_pct": round(100.0 * float((grips < 0).mean()), 1),
            "opening_start": round(start, 4),
            "opening_end": round(float(trace[-1]), 4),
            "opening_min": round(float(np.min(trace)), 4),
            # Approach should be open; a command to close over an untouched pile
            # is the symptom of the wrong sign convention for this checkpoint.
            "sign_suspect": bool(grips[0] < 0.0 and start > 0.9),
        }
    finally:
        env.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=env_factory.DEFAULT_CONFIG)
    ap.add_argument("--model", default=None, choices=["raw", "lora"])
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--steps", type=int, default=40,
                    help="model-driven steps in part C")
    ap.add_argument("--out", default="outputs/b0")
    args = ap.parse_args()

    global CFG
    CFG = args.config

    out = {"seed": args.seed, "servo": run_servo(args.seed), "adapter": run_adapter()}
    print(f"[b0] servo: +1 -> opening {out['servo']['open']['opening_end']:.3f}, "
          f"-1 -> {out['servo']['close']['opening_end']:.3f} "
          f"(ok={out['servo']['ok']})", flush=True)
    print(f"[b0] adapter: LIBERO +1 -> a3 {out['adapter']['libero_close']['a3_grip_dim6']}, "
          f"LIBERO -1 -> a3 {out['adapter']['libero_open']['a3_grip_dim6']} "
          f"(ok={out['adapter']['ok']})", flush=True)

    if args.model:
        out["checkpoint"] = run_checkpoint(args.model, args.seed, args.steps)
        c = out["checkpoint"]
        print(f"[b0] {args.model}: mode={c['gripper_mode']} "
              f"raw chunk grip [{c['raw_chunk_grip']['min']}, {c['raw_chunk_grip']['max']}] "
              f"adapted<0 {c['adapted_grip_minus1_pct']}% "
              f"opening {c['opening_start']:.3f} -> {c['opening_end']:.3f} "
              f"(min {c['opening_min']:.3f}) sign_suspect={c['sign_suspect']}",
              flush=True)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "b0.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"[b0] wrote {path}")

    if not (out["servo"]["ok"] and out["adapter"]["ok"]):
        raise SystemExit("[b0] servo/adapter sign check FAILED; do not run closed loop yet")


if __name__ == "__main__":
    main()
