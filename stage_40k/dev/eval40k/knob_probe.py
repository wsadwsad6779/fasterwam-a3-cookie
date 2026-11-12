"""Measure what each scene-randomisation knob actually delivers, on CPU, with no model in the loop.

The batch eval has to state honestly whether its 20 episodes per node are 20 different scenes or
one scene replayed 20 times. `seed_screen.py` already answered that for the eval defaults: one.
This script answers the next question, which is whether *any* knob can buy real layout diversity
without buying invalid scenes, and it reports the size of the change in millimetres, because a
"diversity split" that moves a cookie 0.05 mm is not a split a policy can react to.

Read from the env source, the knobs are not one thing:

* `position_noise_m` / `yaw_noise_rad` (`make_env`, applied by `_sample_source_cookie_pose`) draw
  an **independent** offset per cookie, and only when `reset` gets `randomize_cookies=True`;
* `randomize_boxes` (reset option) translates the whole source bin by up to +/-1 mm and re-seats
  the target bin by up to +/-2 mm and +/-0.015 rad, so the batch keeps its internal arrangement;
* `batch_offset_m` (`env_factory.apply_batch_offset`) translates all 80 cookies by one shared
  vector drawn from `env.np_random`, which is why it is reproducible from the seed.

So per-cookie noise is the knob the source bin's packing forbids (0.40 mm column pitch, 2.50 mm
row pitch), and the two shared-translation knobs are the ones that can move the layout without
interpenetrating. Which of them is actually valid, and how many distinct scenes each seed really
produces, is measured below rather than assumed. Nothing here touches the GPU or a checkpoint.

The measurement that motivated this script is the reason the reset-time hash is not enough. At
`position_noise_m=0.025` the draws are genuinely applied -- all 80 `_sample_source_cookie_pose`
calls return non-zero offsets and `set_cookie_pose` writes them -- and yet seed 11 ends up with
every cookie back at its nominal slot to the last printed digit, i.e. the packed bin's contact
solver squeezes the batch back into the grid the config describes, while seeds 12/13/14
interpenetrate too hard to recover and trip `Nan, Inf or huge value in QACC at DOF 501`. A seed
whose scene is randomised and then un-randomised by physics is E0 with extra steps, so every cell
here is measured twice: at frame 0, and again after the bin has been allowed to settle.

    cd /workspace/_v1_stage/dev/eval40k
    PYTHONDONTWRITEBYTECODE=1 nice -n 19 python knob_probe.py --seeds 11 --count 8
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

from force_stream import select_channels  # noqa: E402
from seed_screen import build_env_factory, scene_fingerprint  # noqa: E402

FORCE_KEY = "observation.force"
# randomization_metadata entries that describe where things are, as opposed to what they look like.
GEOMETRY_METADATA_KEYS = {"cookie_positions", "cookie_quaternions", "source_bin_position",
                          "target_bin_position", "target_bin_quaternion", "batch_offset_xy_m"}

CELLS = [
    ("E0_eval_defaults", {}),
    ("boxes_jitter", {"randomize_boxes": True}),
    ("batch_offset_8mm", {"batch_offset_m": 0.008}),
    ("batch_offset_10mm", {"batch_offset_m": 0.010}),
    ("batch_offset_20mm", {"batch_offset_m": 0.020}),
    ("batch_offset_50mm", {"batch_offset_m": 0.050}),
    ("cookie_noise_0.3mm", {"mkw": {"position_noise_m": 0.0003}}),
    ("cookie_noise_1mm", {"mkw": {"position_noise_m": 0.001}}),
    ("cookie_noise_25mm", {"mkw": {"position_noise_m": 0.025}}),
    ("boxes_and_offset_20mm", {"randomize_boxes": True, "batch_offset_m": 0.020}),
]


def _hash_qpos(uw, joint_names):
    import mujoco
    m = uw.model
    adr = sorted(int(m.jnt_qposadr[i]) for i in range(m.njnt)
                 if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) or "") in joint_names)
    if not adr:
        return None
    q = np.round(np.asarray(uw.data.qpos, dtype=np.float64)[adr], 7)
    if q.size == 0 or not np.all(np.isfinite(q)):
        return None
    return hashlib.sha256(np.ascontiguousarray(q).tobytes()).hexdigest()[:16]


def target_bin_hash(uw):
    """Hash of the target bin's free joint, which `randomize_boxes` re-seats and cookies do not."""
    return _hash_qpos(uw, {"target_bin_free"})


def appearance_hash(uw):
    """Hash of `randomization_metadata` minus its geometry entries: does the seed change the look?"""
    md = getattr(uw, "randomization_metadata", None)
    if not isinstance(md, dict):
        return None
    keep = {k: md[k] for k in sorted(md) if k not in GEOMETRY_METADATA_KEYS}
    if not keep:
        return None
    return hashlib.sha256(repr(keep).encode("utf-8")).hexdigest()[:16]


def probe_one(env_factory, cell_name, cell, seed, settle_steps=50):
    """Reset, read the scene, then let it run `settle_steps` of bare physics and read it again.

    The second reading is the one that matters. Measured here: at 0.025 m per-cookie noise the
    draws happen (all 80 `_sample_source_cookie_pose` calls return non-zero offsets and
    `set_cookie_pose` writes them), and for seed 11 the contact solver of the packed bin then
    pushes every cookie back to exactly its nominal slot, so the scene the policy is shown is E0
    and the hash confirms it -- while seeds 12/13/14 interpenetrate hard enough that the solver
    cannot recover and the sim reports `Nan, Inf or huge value in QACC at DOF 501`. A knob that
    survives `reset` but not the first 2.5 s of the episode is not a diversity split, so each cell
    reports both states and the drift between them.
    """
    env = env_factory.make_env(render_cameras=False, **(cell.get("mkw") or {}))
    row = {"cell": cell_name, "seed": int(seed), "knobs": {k: v for k, v in cell.items()
                                                           if k != "mkw"},
           "make_env": cell.get("mkw") or {}}
    try:
        obs, info = env_factory.reset(
            env, int(seed),
            randomize_cookies=cell.get("randomize_cookies", True),
            randomize_boxes=cell.get("randomize_boxes", False),
            batch_offset_m=cell.get("batch_offset_m", 0.0))
        uw = getattr(env, "unwrapped", env)
        sanity = env_factory.scene_sanity(env)
        nominal = np.asarray(uw.SOURCE_POSITIONS, dtype=np.float64)[:, :2]
        actual = np.asarray(uw.cookie_positions, dtype=np.float64)[:, :2]
        shift_mm = np.linalg.norm(actual - nominal, axis=1) * 1000.0
        f18 = np.asarray(obs[FORCE_KEY], dtype=np.float64)
        row.update({
            "sanity_ok": bool(sanity["ok"]),
            "max_penetration_mm": float(sanity["max_penetration_mm"]),
            "worst_pair": [int(x) for x in sanity["worst_pair"]],
            "cookies_outside_bin": int(sanity["cookies_outside_bin"]),
            "finite": bool(sanity["finite"]),
            "geo_hash": scene_fingerprint(uw),
            "target_bin_hash": target_bin_hash(uw),
            "appearance_hash": appearance_hash(uw),
            "cookie_shift_from_nominal_mm": {"max": round(float(shift_mm.max()), 3),
                                             "mean": round(float(shift_mm.mean()), 3),
                                             "n_nonzero": int(np.count_nonzero(shift_mm > 1e-9))},
            "nominal_matches_sim": bool(np.allclose(actual, nominal, atol=1e-9)),
            "cookies_in_source": info.get("cookies_in_source"),
            "required_cookies": info.get("required_cookies"),
            "batch_offset_xy_m": [round(float(x), 5) for x in
                                  (info.get("batch_offset_xy_m") or [0.0, 0.0])],
            "force6_newton": [round(float(x), 4) for x in select_channels(f18)],
        })
        if settle_steps:
            import mujoco
            for _ in range(int(settle_steps)):
                mujoco.mj_step(uw.model, uw.data)
            mujoco.mj_forward(uw.model, uw.data)
            after = np.asarray(uw.cookie_positions, dtype=np.float64)[:, :2]
            drift = np.linalg.norm(after - actual, axis=1) * 1000.0
            rest = np.linalg.norm(after - nominal, axis=1) * 1000.0
            s2 = env_factory.scene_sanity(env)
            row["after_steps"] = {
                "steps": int(settle_steps),
                "sim_s": round(int(settle_steps) * float(uw.model.opt.timestep), 5),
                "drift_mm": {"max": round(float(drift.max()), 3),
                             "mean": round(float(drift.mean()), 3)},
                "shift_from_nominal_mm": {"max": round(float(rest.max()), 3),
                                          "mean": round(float(rest.mean()), 3),
                                          "n_nonzero": int(np.count_nonzero(rest > 1e-9))},
                "geo_hash": scene_fingerprint(uw),
                "target_bin_hash": target_bin_hash(uw),
                "sanity_ok": bool(s2["ok"]),
                "max_penetration_mm": float(s2["max_penetration_mm"]),
                "cookies_outside_bin": int(s2["cookies_outside_bin"]),
                "qacc_unstable": bool(not np.all(np.isfinite(uw.data.qacc))),
            }
    except Exception as exc:
        row["error"] = "%s: %s" % (type(exc).__name__, exc)
    finally:
        if hasattr(env, "close"):
            env.close()
    return row


def valid(row, after=False):
    """Frame-0 validity, or validity after the settle steps.

    `cookies_in_source` comes from `reset`'s info and is only read from there; the post-settle
    state is judged by `scene_sanity`, which recomputes how many cookies left the bin.
    """
    if row.get("error"):
        return False
    src = row.get("after_steps") if after else row
    if src is None:
        return False
    return (src.get("sanity_ok") and src.get("cookies_outside_bin") == 0
            and row.get("cookies_in_source") not in (0, None))


def summarise(rows, ref_geo=None):
    geos = [r.get("geo_hash") for r in rows if r.get("geo_hash")]
    tgts = [r.get("target_bin_hash") for r in rows if r.get("target_bin_hash")]
    apps = [r.get("appearance_hash") for r in rows if r.get("appearance_hash")]
    ok_rows = [r for r in rows if valid(r)]
    shifts = [r["cookie_shift_from_nominal_mm"]["max"] for r in rows
              if r.get("cookie_shift_from_nominal_mm")]
    same = sum(1 for r in rows if ref_geo and r.get("geo_hash") == ref_geo)
    aft = [r.get("after_steps") for r in rows if r.get("after_steps")]
    aft_geos = [a["geo_hash"] for a in aft if a.get("geo_hash")]
    aft_ok = sum(1 for r in rows if valid(r, after=True))
    aft_same = sum(1 for a in aft_geos if ref_geo and a == ref_geo)
    return {"cell": rows[0]["cell"], "seeds": len(rows), "valid": len(ok_rows),
            "distinct_geo": len({g for g in geos}), "distinct_geo_valid": len(
                {r.get("geo_hash") for r in ok_rows if r.get("geo_hash")}),
            "identical_to_reference_layout": same,
            "distinct_target_bin": len({t for t in tgts}),
            "distinct_appearance": len({a for a in apps}),
            "worst_penetration_mm": round(max([r.get("max_penetration_mm", 0.0)
                                               for r in rows] or [0.0]), 3),
            "median_max_shift_mm": (float(np.median(shifts)) if shifts else None),
            "after_settle": {"valid": aft_ok, "distinct_geo": len({g for g in aft_geos}),
                             "identical_to_reference_layout": aft_same,
                             "median_max_shift_mm": (float(np.median(
                                 [a["shift_from_nominal_mm"]["max"] for a in aft])) if aft
                                 else None),
                             "median_drift_mm": (float(np.median(
                                 [a["drift_mm"]["max"] for a in aft])) if aft else None)},
            "errors": [r.get("error") for r in rows if r.get("error")][:2]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=11)
    ap.add_argument("--count", type=int, default=8)
    ap.add_argument("--cells", nargs="*", default=None,
                    help="subset of the cell names; default runs the whole grid")
    ap.add_argument("--out", default=str(STAGE / "dev" / "out40k" / "knob_probe.json"))
    ap.add_argument("--settle-steps", type=int, default=50,
                   help="bare mj_step calls run after reset to see whether the randomisation "
                        "survives physics; 0 skips that half of the measurement")
    args = ap.parse_args()

    names = {c[0] for c in CELLS}
    wanted = args.cells or sorted(names)
    unknown = [c for c in wanted if c not in names]
    if unknown:
        raise SystemExit("unknown cell(s) %s; available: %s" % (unknown, sorted(names)))
    seeds = list(range(args.seeds, args.seeds + args.count))
    env_factory = build_env_factory()

    rows, summaries = [], []
    ref_geo = None
    for cell_name, cell in CELLS:
        if cell_name not in wanted:
            continue
        cell_rows = [probe_one(env_factory, cell_name, cell, s,
                               settle_steps=args.settle_steps) for s in seeds]
        if cell_name == "E0_eval_defaults" and ref_geo is None:
            ref_geo = cell_rows[0].get("geo_hash")     # the layout the corpus was recorded on
        rows.extend(cell_rows)
        sm = summarise(cell_rows, ref_geo)
        summaries.append(sm)
        print("%-22s valid=%2d/%2d geo=%2d same_as_E0=%2d target=%2d look=%2d pen=%7.3fmm "
              "shift=%s" %
              (cell_name, sm["valid"], sm["seeds"], sm["distinct_geo"],
               sm["identical_to_reference_layout"], sm["distinct_target_bin"],
               sm["distinct_appearance"], sm["worst_penetration_mm"],
               ("%.3fmm" % sm["median_max_shift_mm"]) if sm["median_max_shift_mm"] is not None
               else "n/a"), flush=True)
        for r in cell_rows:
            a = r.get("after_steps") or {}
            print("   seed=%-5d ok=%-5s pen=%7.3f outside=%s in_source=%s geo=%s tgt=%s look=%s"
                  " shift=%s same_as_E0=%s %s" %
                  (r["seed"], r.get("sanity_ok"), r.get("max_penetration_mm", -1),
                   r.get("cookies_outside_bin"), r.get("cookies_in_source"),
                   str(r.get("geo_hash"))[:8], str(r.get("target_bin_hash"))[:8],
                   str(r.get("appearance_hash"))[:8],
                   json.dumps(r.get("cookie_shift_from_nominal_mm")),
                   r.get("geo_hash") == ref_geo if ref_geo else "n/a",
                   str(r.get("error", ""))[:60]), flush=True)
            if a:
                print("      after %d mj_step: ok=%-5s pen=%7.3f outside=%s geo=%s same_as_E0=%s "
                      "drift=%s shift=%s qacc_bad=%s" %
                      (a["steps"], a["sanity_ok"], a["max_penetration_mm"],
                       a["cookies_outside_bin"], str(a["geo_hash"])[:8],
                       a["geo_hash"] == ref_geo if ref_geo else "n/a",
                       json.dumps(a["drift_mm"]), json.dumps(a["shift_from_nominal_mm"]),
                       a["qacc_unstable"]), flush=True)

    doc = {"seeds": seeds, "count": len(seeds), "reference_layout_geo": ref_geo,
           "cells": summaries, "rows": rows,
            "definitions": {
                "valid": "scene_sanity.ok, no cookie outside the source bin, and the source bin "
                        "not empty at frame 0",
                "distinct_geo": "different `cookie_*_free` qpos hashes across ALL seeds, "
                                "including invalid scenes",
                "distinct_geo_valid": "different layouts among the scenes that are also valid",
                "distinct_appearance": "different non-geometry `randomization_metadata` "
                                       "(textures/lighting), which changes the pixels but not the "
                                       "task",
                "shift": "how far a cookie sits from its nominal configured slot, in mm",
                "after_settle": "the same measurements after `settle_steps` bare `mj_step` calls "
                                "with no action applied -- `sim_s` says how much simulated time "
                                "that is at `model.opt.timestep`, and it is NOT counted in "
                                "episode frames because one env.step wraps an unknown number of "
                                "solver substeps; `drift` is how far a cookie moved during those "
                                "steps, and `identical_to_reference_layout` counts the seeds "
                                "whose post-settle layout is E0 again",
                "knob_source": "read from a3_dual_arm_sim/tasks/cookie_transfer.py reset(): "
                               "per-cookie noise needs randomize_cookies and uses "
                               "task_config.position_noise_m; randomize_boxes jitters the source "
                               "bin +/-1 mm and the target bin +/-2 mm and +/-0.015 rad; "
                               "batch_offset_m is one shared vector for all 80 cookies"}}
    out = Path(args.out)
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(doc, indent=2, sort_keys=True, default=str))
    print("\nverdict per cell, judged on the state AFTER the settle steps (what an episode "
          "actually starts from):")
    for s in summaries:
        a = s["after_settle"]
        if not a["distinct_geo"]:
            print("  %s: no post-settle measurement taken (--settle-steps 0)" % s["cell"])
            continue
        if a["valid"] != s["seeds"]:
            print("  %-22s rejected: only %d/%d seeds are still valid after the settle steps"
                  % (s["cell"], a["valid"], s["seeds"]))
            continue
        if a["distinct_geo"] < 2:
            print("  %-22s rejected: %d/%d seeds settle back into one and the same layout"
                  % (s["cell"], a["identical_to_reference_layout"], s["seeds"]))
            continue
        print("  %-22s %d distinct post-settle layouts (%d of them are E0), median shift %.3f mm,"
              " median drift during those steps %.3f mm -- %s"
              % (s["cell"], a["distinct_geo"], a["identical_to_reference_layout"],
                 a["median_max_shift_mm"], a["median_drift_mm"],
                 "usable as a diversity split" if a["median_max_shift_mm"] >= 1.0 else
                 "sub-millimetre: a different hash, not a different task"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
