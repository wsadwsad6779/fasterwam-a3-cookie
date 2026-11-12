"""Screen eval seeds against the real scene generator, on CPU, with no model in the loop.

The batch eval at the end of the 40k run spends 20 episodes per node on one fixed seed list, one
episode per seed, so a seed whose scene is invalid is not a failed episode -- it is a hole in every
node's denominator.
MuJoCo runs on CPU, so this can be done now: reset the scene, run `scene_sanity`, and record what
the starting state actually looks like. Nothing here touches the GPU, loads a checkpoint, or steps
the robot.

Three properties are checked rather than assumed:

* validity -- `scene_sanity(env)["ok"]`, the same gate `eval40k.py` refuses to evaluate past;
* determinism -- a second pass over the chosen seeds must reproduce the same verdict, the same
  starting numbers and the same scene hash. A seed that is only reproducible within one process
  is not a benchmark seed.
* diversity -- how many *different* scenes the chosen seeds actually produce. This is measured
  because the first run of this script found the answer is one: with `make_env`'s default
  `position_noise_m=0.0`, `yaw_noise_rad=0.0` and `randomize_boxes=False`, `reset(seed)` draws
  nothing and every seed starts from the identical cookie layout. Measured over five seeds, the
  target bin pose and the recorded appearance metadata are just as identical, so at the eval
  defaults a seed changes neither the geometry nor the pixels -- only the policy's own diffusion
  draw. `knob_probe.py` is the follow-up that runs every randomisation knob in the env and says
  what each one is worth; passing `--position-noise` here still prescreens one split, and
  `scene_sanity` is what rejects the interpenetrating scenes too much noise creates.

    cd /workspace/_v1_stage/dev/eval40k
    PYTHONDONTWRITEBYTECODE=1 nice -n 19 python seed_screen.py --want 20 --scan 60
    PYTHONDONTWRITEBYTECODE=1 nice -n 19 python seed_screen.py --want 20 --scan 60 \
        --position-noise 0.025
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
STAGE = HERE.parent.parent
sys.path.insert(0, str(HERE))

from force_stream import CHANNEL_NAMES, select_channels  # noqa: E402

BRIDGE_DIRS = (str(Path("/workspace/fasterwam_a3_bridge/bridge")),
               str(Path("/workspace/fasterwam_a3_bridge/bridge/_next")),
               str(Path("/workspace/a3_dual_arm_sim/src")))
FORCE_KEY = "observation.force"


def build_env_factory():
    for p in BRIDGE_DIRS:
        if p not in sys.path:
            sys.path.insert(0, p)
    import env_factory
    return env_factory


def scene_fingerprint(uw):
    """Hash of the cookies' start pose, or None if this env build has no cookie joints.

    The `cookie_*_free` qpos slice is hashed rather than `unwrapped.cookie_positions` because that
    property reads `data.xpos`, which is only current after a forward pass, while qpos is the
    state itself. Both do move with the randomisation knobs -- `knob_probe.py` compares them.
    """
    try:
        import mujoco
        m = uw.model
        adr = sorted(int(m.jnt_qposadr[i]) for i in range(m.njnt)
                     if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) or "")
                     .startswith("cookie_"))
        if not adr:
            return None
        q = np.round(np.asarray(uw.data.qpos, dtype=np.float64)[adr], 7)
    except Exception:
        return None
    if q.size == 0 or not np.all(np.isfinite(q)):
        return None
    return hashlib.sha256(np.ascontiguousarray(q).tobytes()).hexdigest()[:16]


def screen_one(env_factory, seed, randomize_boxes=False, mkw=None):
    """Reset seed `seed` once and report what the scene and sensors say at frame 0."""
    env = env_factory.make_env(render_cameras=False, **(mkw or {}))
    out = {"seed": int(seed)}
    try:
        obs, info = env_factory.reset(env, int(seed), randomize_cookies=True,
                                      randomize_boxes=randomize_boxes)
        sanity = env_factory.scene_sanity(env)
        uw = getattr(env, "unwrapped", env)
        f18 = np.asarray(obs[FORCE_KEY], dtype=np.float64)
        pose = np.asarray(obs.get("observation.eef_pose", []), dtype=np.float64)
        out.update({
            "sanity_ok": bool(sanity["ok"]),
            "sanity": {k: v for k, v in sanity.items() if k != "ok"},
            "force18_finite": bool(np.all(np.isfinite(f18))),
            "force6_newton": [round(float(x), 4) for x in select_channels(f18)],
            "start_contact_max_abs_n": round(float(np.max(np.abs(f18))), 4)
            if f18.size else None,
            "eef_pose_finite": bool(pose.size == 0 or np.all(np.isfinite(pose))),
            "cookies_in_target": info.get("cookies_in_target"),
            "cookies_to_place": info.get("cookies_to_place"),
            "cookies_in_source": info.get("cookies_in_source"),
            "required_cookies": info.get("required_cookies"),
            "info_keys": sorted(str(k) for k in info),
            "scene_fingerprint": scene_fingerprint(uw),
        })
    except Exception as exc:                      # a bad seed must not lose the whole scan
        out["error"] = "%s: %s" % (type(exc).__name__, exc)
    finally:
        if hasattr(env, "close"):
            env.close()
    return out


def valid(row):
    if row.get("error"):
        return False
    return (row["sanity_ok"] and row["force18_finite"] and row["eef_pose_finite"]
            and row.get("cookies_in_target") in (0, None))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--want", type=int, default=20, help="valid seeds to collect")
    ap.add_argument("--scan", type=int, default=60, help="candidate seeds to try")
    ap.add_argument("--start", type=int, default=11)
    ap.add_argument("--randomize-boxes", action="store_true")
    ap.add_argument("--position-noise", type=float, default=0.0,
                   help="object position noise handed to make_env; 0 means every seed gets the "
                        "same scene, so a diversity split needs a non-zero value here")
    ap.add_argument("--yaw-noise", type=float, default=0.0)
    ap.add_argument("--recheck", type=int, default=3,
                    help="re-run this many of the chosen seeds in a second pass to prove the "
                         "scene reproduces; 0 skips the determinism pass")
    ap.add_argument("--out", default=str(STAGE / "dev" / "out40k" / "seed_screen.json"))
    args = ap.parse_args()
    mkw = {"position_noise_m": args.position_noise, "yaw_noise_rad": args.yaw_noise}

    env_factory = build_env_factory()
    rows = []
    chosen = []
    for seed in range(args.start, args.start + args.scan):
        row = screen_one(env_factory, seed, randomize_boxes=args.randomize_boxes, mkw=mkw)
        rows.append(row)
        take = valid(row) and len(chosen) < args.want
        if take:
            chosen.append(seed)
        print("seed=%-5d ok=%-5s contact_max=%8s N in_target=%s %s%s"
              % (seed, row["sanity_ok"], row["start_contact_max_abs_n"],
                 row["cookies_in_target"], "TAKEN" if take else "",
                 " " + str(row.get("error", ""))[:80], ), flush=True)
        if len(chosen) >= args.want:
            break

    unstable = []
    for seed in chosen[:max(0, args.recheck)]:
        first = next(r for r in rows if r["seed"] == seed)
        again = screen_one(env_factory, seed, randomize_boxes=args.randomize_boxes, mkw=mkw)
        same = (valid(again) == valid(first)
                and again["force6_newton"] == first["force6_newton"]
                and again.get("scene_fingerprint") == first.get("scene_fingerprint"))
        if not same:
            unstable.append(seed)
        print("recheck seed=%-5d reproduced=%s" % (seed, same), flush=True)

    fps = [r.get("scene_fingerprint") for r in rows if r["seed"] in chosen]
    n_distinct = len({f for f in fps if f})
    diversity = {"chosen": len(chosen), "fingerprints_recorded": len(fps),
                 "distinct_scenes": n_distinct,
                 "usable": bool(fps and n_distinct > 1),
                 "make_env": mkw, "randomize_boxes": args.randomize_boxes}

    doc = {"want": args.want, "scanned": len(rows), "valid_seeds": chosen,
           "n_valid": len(chosen), "unreproducible": unstable,
           "scene_diversity": diversity,
           "criteria": ["scene_sanity.ok", "force18 finite", "eef pose finite",
                        "no cookie already in target at frame 0"],
           "randomize_boxes": args.randomize_boxes, "rows": rows}
    out = Path(args.out)
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(doc, indent=2, sort_keys=True, default=str))
    print(json.dumps({k: doc[k] for k in ("want", "scanned", "n_valid", "valid_seeds",
                                          "unreproducible")}, indent=2))
    print("scene diversity:", json.dumps(diversity))
    if len(chosen) < args.want or unstable:
        print("NOTE the list is short or a seed did not reproduce; the eval seed set is not "
              "fixed yet", file=sys.stderr)
        return 1
    if not diversity["usable"]:
        print("NOTE the scene positions are not recorded or are identical across seeds, so this "
              "run proves the 20 seeds are legal and reproducible but not that they are 20 "
              "different scenes", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
