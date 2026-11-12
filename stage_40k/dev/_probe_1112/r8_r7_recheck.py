"""Would round 8's split check have passed the round-7 episodes it replaced?

Round 8 changed an assertion, and the only way to show that the change raised the standard instead
of lowering it is to run the new standard against the old evidence. `out/smoke_20261112_r7` is the
run that failed 45/47, and its JSON carries every field `proprio_band_rollup` needs -- per-column
normalised extremes, the raw witness of the same replan with its frame, the clamped-call counts.
So the replay is pure CPU arithmetic on a file that already exists: no model, no GPU, no simulator,
and nothing written except this report.

Reads the round-7 JSON and, for each episode, prints what round 7 asserted (the two +-1.02
aggregate bounds, which were false), what round 8's interface terms say (true, and they were true in
round 7 too), and what round 8's attribution terms require of the same numbers. Exits non-zero if
the attribution does not fully explain the round-7 out-of-band columns, which is the case where the
new check would be hiding the old failure rather than naming it.

Two ways this script could produce a false clean bill, both discovered by running it:

* `proprio_band_rollup` returns an empty, all-zero rollup when the stats band was never loaded,
  which reads exactly like "no episode left the band". The band is therefore loaded here from the
  same file the driver loads, required to be present, and required to equal the band the round-7
  run itself recorded;
* `all(...)` over a dict that also carries measurements is false for reasons that have nothing to
  do with the policy, which reads exactly like "round 8 would still fail". Only booleans go in the
  verdict; numbers are reported beside them.
"""
import json
import sys
from pathlib import Path

import eval40k as E
import smoke40k as S

R7 = Path("/workspace/_v1_stage/dev/eval40k/out/smoke_20261112_r7/step_040000"
          "/smoke_seeds11_12.json")
OUT = Path("/workspace/_v1_stage/dev/out40k/podrun/r8_r7_recheck.json")


def load_band_the_way_the_driver_does():
    """Fill `eval40k.STATS_FACTS` so `stats_band_vectors()` answers, and prove it is the right band.

    The live run fills it inside `load_model_and_policy`, from `--stats` and the calibration file;
    this replay never opens a model, so it calls the same `check_stats_width` the driver calls. The
    band the round-7 JSON recorded is the third opinion on the same file, and all three must agree
    before the replay's silence about excursions could mean anything.
    """
    calib = E.read_json(E.CALIB_PATH)
    from fasterwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
    stats = load_dataset_stats_from_json(str(E.V1G_STATS))
    E.check_stats_width(stats, str(E.V1G_STATS), calib_std=calib["std"])
    band = S.stats_band_vectors()
    if band is None:
        raise SystemExit("replay aborted: stats_band_vectors() returned None even after loading "
                         "%s -- an empty rollup here would read as 'the robot stayed in band'"
                         % E.V1G_STATS)
    return {"source": E.STATS_FACTS.get("source"), "lo": [float(x) for x in band[0]],
            "hi": [float(x) for x in band[1]]}


def main():
    res = json.loads(R7.read_text())
    band = load_band_the_way_the_driver_does()
    recorded = (res.get("proprio_band") or {})
    same = (recorded.get("lo") == band["lo"] and recorded.get("hi") == band["hi"])
    rep = {"round7_json": str(R7), "round7_summary": res["summary"],
           "band_replayed": band, "band_recorded_by_round7": {
               "source": recorded.get("source"), "lo": recorded.get("lo"),
               "hi": recorded.get("hi")},
           "replay_band_equals_run_band": same,
           "round7_failed_checks": [c["name"] for c in res["checks"] if not c["ok"]],
           "episodes": []}
    if not same:
        raise SystemExit("replay aborted: the band loaded now differs from the band the round-7 "
                         "run recorded, so its exceedances are not the ones being replayed")
    honest = True
    for rec in res["episodes"]:
        ood = S.episode_band_finding(rec)
        r7_terms = {"base_min": rec["proprio_base_min_all"], "base_max": rec["proprio_base_max_all"],
                    "in_102_old_bound": (-1.02 <= rec["proprio_base_min_all"] <= 1.02
                                         and -1.02 <= rec["proprio_base_max_all"] <= 1.02)}
        # The two frozen deployment values are compared literally here, where the live check reads
        # them off the config: this script has no model to ask, and `flow_test` is what proves the
        # live check reads rather than assumes.
        r8_interface = {
            "no_gradient_output": not rec["action_requires_grad_any"],
            "deployment_diffusion_steps": rec["num_inference_steps_seen"] == [10],
            "rand_device_as_deployed": rec["rand_device_seen"] == ["cpu"],
            "proprio_14_wide": rec["proprio_widths"] == [14],
            "force_clamp_held": (abs(rec["proprio_force_min_all"]) <= 5.05
                                 and abs(rec["proprio_force_max_all"]) <= 5.05)}
        r8_attribution = {
            "unexplained_zero": ood["unexplained"] == 0,
            "nothing_unexplained": ood["any_excursion_unexplained"] is False,
            "witness_present": ood["episodes_without_witness"] == [],
            "band_map_paired_every_replan": (rec.get("proprio_band_map_pairs")
                                             == rec.get("infer_action_calls")
                                             and rec.get("proprio_band_map_unpaired") == 0)}
        r8_measurements = {
            "band_map_maxdiff": rec.get("proprio_band_map_maxdiff"),
            "band_map_pairs": rec.get("proprio_band_map_pairs"),
            "band_map_unpaired": rec.get("proprio_band_map_unpaired"),
            "infer_action_calls": rec.get("infer_action_calls")}
        # The point of the replay: the episode really did leave the band, the new check has to say
        # so, and it has to say *why*. A check that passed because the columns looked in-band would
        # be the weakening this round is required not to do.
        leaves_band = bool(ood["base_columns_out_of_band"] or ood["force_columns_out_of_band"])
        rep["episodes"].append({
            "seed": rec["seed"], "round7_assertion_terms": r7_terms,
            "round8_interface_terms": r8_interface, "round8_attribution_terms": r8_attribution,
            "round8_measurements": r8_measurements,
            "episode_band_finding": ood, "episode_leaves_the_band": leaves_band,
            "check_would_pass": all(r8_interface.values()) and all(r8_attribution.values()),
            "standard_raised_not_lowered": leaves_band and ood["unexplained"] == 0
                                           and ood["clamped_calls"] > 0})
        honest = honest and rep["episodes"][-1]["check_would_pass"]
        honest = honest and rep["episodes"][-1]["standard_raised_not_lowered"]
    rep["all_round8_checks_would_pass_on_round7_data"] = honest
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, indent=2, sort_keys=True, default=str))
    print(json.dumps({"out": str(OUT),
                      "band": band["source"],
                      "replay_band_equals_run_band": same,
                      "episodes": [e["seed"] for e in rep["episodes"]],
                      "would_pass": [e["check_would_pass"] for e in rep["episodes"]],
                      "leaves_band": [e["episode_leaves_the_band"] for e in rep["episodes"]],
                      "clamped_calls": [e["episode_band_finding"]["clamped_calls"]
                                        for e in rep["episodes"]],
                      "unexplained": [e["episode_band_finding"]["unexplained"]
                                      for e in rep["episodes"]],
                      "worst_base_widths": [e["episode_band_finding"][
                          "worst_base_excursion_band_widths"] for e in rep["episodes"]],
                      "all_round8_checks_would_pass_on_round7_data": honest}, indent=2))
    return 0 if honest else 1


if __name__ == "__main__":
    sys.exit(main())
