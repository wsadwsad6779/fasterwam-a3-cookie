"""CPU-only seed pre-screen for the reconstructed-normalizer 40k diagnosis (Req4 pre-flight).

Reuses seed_screen.screen_one / valid BY IDENTITY (the original protocol's scene gate), but wires
env_factory through the RESTORED paths instead of seed_screen.py's dead historical BRIDGE_DIRS.
No model, no GPU, no checkpoint, no normalizer: pure MuJoCo reset + scene_sanity, exactly the gate
eval40k refuses to evaluate past. Picks valid seeds for the Req4 shortest block (11/17 reserved for
Req6) and confirms 11/17 are themselves valid scenes under the same gate.

At the eval defaults (position_noise_m=0, yaw_noise_rad=0, randomize_boxes=False) every seed draws
the identical cookie layout, so a seed changes only the policy's diffusion draw, not the scene --
this is measured here via scene_fingerprint, not assumed.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

BASE = Path(os.environ.get("EVAL_RESTORE_BASE", "/workspace/eval_restore_20261117"))
EVAL40K = BASE / "deploy" / "stage_40k" / "dev" / "eval40k"
BRIDGE = BASE / "deploy" / "bridge"
BRIDGE_NEXT = BRIDGE / "_next"
SIM_SRC = BASE / "deploy" / "a3_dual_arm_sim" / "src"
SIM_CFG = BASE / "deploy" / "a3_dual_arm_sim" / "configs" / "cookie_batch.yaml"


def main():
    sys.path.insert(0, str(EVAL40K))
    for p in (str(BRIDGE), str(BRIDGE_NEXT), str(SIM_SRC)):
        if p not in sys.path:
            sys.path.insert(0, p)
    os.environ["A3_CONFIG_PATH"] = str(SIM_CFG)
    import env_factory
    import seed_screen

    mkw = {"position_noise_m": 0.0, "yaw_noise_rad": 0.0}
    chosen, rows = [], []
    for seed in range(11, 71):
        if seed in (11, 17):
            continue
        row = seed_screen.screen_one(env_factory, seed, randomize_boxes=False, mkw=mkw)
        ok = seed_screen.valid(row)
        rows.append({"seed": seed, "valid": ok, "sanity_ok": row.get("sanity_ok"),
                     "contact_max_n": row.get("start_contact_max_abs_n"),
                     "cookies_in_target": row.get("cookies_in_target"),
                     "scene_fingerprint": row.get("scene_fingerprint"),
                     "error": row.get("error")})
        if ok and len(chosen) < 5:
            chosen.append(seed)
        if len(chosen) >= 5:
            break

    req6 = {}
    for seed in (11, 17):
        row = seed_screen.screen_one(env_factory, seed, randomize_boxes=False, mkw=mkw)
        req6[str(seed)] = {"valid": seed_screen.valid(row), "sanity_ok": row.get("sanity_ok"),
                           "contact_max_n": row.get("start_contact_max_abs_n"),
                           "cookies_in_target": row.get("cookies_in_target"),
                           "scene_fingerprint": row.get("scene_fingerprint"),
                           "force6_newton": row.get("force6_newton"),
                           "error": row.get("error")}

    fps = [r.get("scene_fingerprint") for r in rows if r["seed"] in chosen]
    fps += [v.get("scene_fingerprint") for v in req6.values()]
    distinct = len({f for f in fps if f})
    doc = {"NORMALIZER_IDENTITY": "RECONSTRUCTED_DETERMINISTIC_NOT_ORIGINAL",
           "original_stats_sha256": "UNKNOWN",
           "scope": "Req4 CPU seed pre-flight; pure MuJoCo scene_sanity, no model/GPU/normalizer",
           "sim_config": str(SIM_CFG), "make_env_kwargs": mkw, "randomize_boxes": False,
           "req4_candidate_seeds": chosen, "screened_rows": rows, "req6_seeds_11_17": req6,
           "distinct_scene_fingerprints": distinct,
           "scene_identical_across_seeds": bool(distinct <= 1),
           "env_factory_file": env_factory.__file__,
           "seed_screen_file": seed_screen.__file__}
    out = BASE / "reconstructed_normalizer" / "seed_preflight.json"
    out.write_text(json.dumps(doc, indent=2, sort_keys=True, default=str))
    print(json.dumps({"req4_candidate_seeds": chosen,
                      "req6_valid": {k: v["valid"] for k, v in req6.items()},
                      "distinct_scene_fingerprints": distinct,
                      "out": str(out)}, indent=2))
    return 0 if chosen else 1


if __name__ == "__main__":
    sys.exit(main())