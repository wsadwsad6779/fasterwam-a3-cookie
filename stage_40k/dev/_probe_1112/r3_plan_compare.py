"""Compare this round's dry-run plan with the protocol frozen on 2026-11-11.

Read-only: it opens two plan.json files and the seed screen, prints a verdict, writes nothing but
its own log. The point is the interface fix must not have moved a protocol knob -- the eight nodes,
the twenty screened seeds, one episode per seed, the horizon, the replan period, the config and the
force calibration have to be byte-identical to what was frozen before the fix, and the only allowed
additions are the three prompt fields the fix introduced.
"""
import json
from pathlib import Path

OUT = Path("/workspace/_v1_stage/dev/eval40k/out")
FROZEN = json.loads((OUT / "dryrun_20261112" / "plan.json").read_text())
NOW = json.loads((OUT / "dryrun_round3" / "plan.json").read_text())
SEEDS = json.loads(Path("/workspace/_v1_stage/dev/out40k/seed_screen.json").read_text())

PROTOCOL = ["backend", "policy", "steps", "seeds", "episodes_per_node", "max_steps", "replan",
            "config", "torch_device_note"]
CALIB = ["version", "calib_version", "channels", "channel_names", "mean", "std", "scale", "units",
         "zero_load_ref_median"]

verdict = {"frozen_generated": FROZEN["generated"], "now_generated": NOW["generated"]}
verdict["top_level"] = {}
for k in PROTOCOL:
    verdict["top_level"][k] = {"frozen": FROZEN.get(k), "now": NOW.get(k),
                               "equal": FROZEN.get(k) == NOW.get(k)}
verdict["calibration"] = {k: {"frozen": FROZEN["calibration"].get(k),
                              "now": NOW["calibration"].get(k),
                              "equal": FROZEN["calibration"].get(k) == NOW["calibration"].get(k)}
                          for k in CALIB}
only_new = sorted(set(NOW) - set(FROZEN))
only_old = sorted(set(FROZEN) - set(NOW))
verdict["keys_only_in_now"] = only_new
verdict["keys_only_in_frozen"] = only_old
verdict["prompt_fields"] = {k: NOW.get(k) for k in
                            ("instruction", "instruction_source", "prompt_cache_file")}
verdict["seeds_are_the_screened_20"] = (NOW["seeds"] == [int(s) for s in SEEDS["valid_seeds"]]
                                        and len(NOW["seeds"]) == 20)
verdict["one_episode_per_seed"] = NOW["episodes_per_node"] == len(NOW["seeds"]) == 20
verdict["eight_nodes_5k_to_40k"] = NOW["steps"] == [5000 * i for i in range(1, 9)]

fb = {n["step"]: n for n in FROZEN["nodes"]}
nb = {n["step"]: n for n in NOW["nodes"]}
verdict["per_node"] = {}
for s in sorted(nb):
    f = fb.get(s, {})
    verdict["per_node"][s] = {
        "frozen_status": f.get("status"), "now_status": nb[s]["status"],
        "frozen_merged": f.get("merged_artifact"), "now_merged": nb[s]["merged_artifact"],
        "now_merged_bytes": nb[s]["merged_bytes"],
        "light_node_equal": f.get("light_node") == nb[s].get("light_node"),
        "global_step_equal": f.get("light_global_step") == nb[s].get("light_global_step"),
        "node_calib_equal": (json.dumps(f.get("light_force_calibration"), sort_keys=True)
                             == json.dumps(nb[s].get("light_force_calibration"), sort_keys=True))}

# the no-merge-allowed invariant: a node whose light dir is present but whose artifact is not must
# be reported as such rather than silently evaluated from the adapter
verdict["statuses_now"] = {s: nb[s]["status"] for s in sorted(nb)}
verdict["statuses_frozen"] = {s: fb.get(s, {}).get("status") for s in sorted(fb)}

problems = []
for k in PROTOCOL + ["calibration." + c for c in CALIB]:
    if k.startswith("calibration."):
        c = k.split(".", 1)[1]
        if not verdict["calibration"][c]["equal"]:
            problems.append(k)
    elif not verdict["top_level"][k]["equal"]:
        problems.append(k)
verdict["protocol_drift"] = problems
verdict["allowed_new_keys"] = sorted(set(only_new))
verdict["unexpected_new_keys"] = sorted(set(only_new) - {"instruction", "instruction_source",
                                                         "prompt_cache_file"})
verdict["all_protocol_fields_equal"] = not problems
verdict["seeds_and_nodes_as_approved"] = bool(verdict["seeds_are_the_screened_20"]
                                             and verdict["one_episode_per_seed"]
                                             and verdict["eight_nodes_5k_to_40k"])
verdict["pass"] = bool(verdict["all_protocol_fields_equal"]
                       and verdict["seeds_and_nodes_as_approved"]
                       and not verdict["unexpected_new_keys"]
                       and not verdict["keys_only_in_frozen"])
print(json.dumps(verdict, indent=1, sort_keys=True, default=str))
