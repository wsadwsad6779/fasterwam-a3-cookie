import json, sys, pathlib
d = pathlib.Path(sys.argv[1])
j = json.loads((d / "adjudication_c3.json").read_text())
s = json.loads((d / "summary.json").read_text())
rows = s["episodes"] if isinstance(s, dict) and "episodes" in s else s
obs = {r.get("seed"): (r.get("anomaly_observed"), (r.get("retry_log") or [{}])[0].get("anomaly_kind")) for r in rows}
for e in j["episodes"]:
    k = obs.get(e["seed"], (None, None))
    ab = e.get("abnormal") or {}
    rc = e.get("recovery") or {}
    out = [str(e["seed"]), "var" + str(e.get("variant")), "noise=" + str(e.get("noise_m")), "driver_anom=" + str(k[0]), "kind=" + str(k[1]), "steps=" + str(e.get("steps")), "cut=" + str(e.get("cut")), "ab_peak=" + str(ab.get("peak")), "ab_over16=" + str(ab.get("over_1p6")), "rec_peak=" + str(rc.get("peak")), "rec_over2=" + str(rc.get("over_2p0")), "no_contact=" + str(e.get("expert_reports_no_contact")), "true_empty=" + str(e.get("true_empty")), "air=" + str(e.get("closed_through_air")), "held_nothing=" + str(e.get("held_nothing")), "hold_rng=" + str(e.get("hold_in_range")), "STRICT=" + str(e.get("strict_accept"))]
    print(" ".join(out))