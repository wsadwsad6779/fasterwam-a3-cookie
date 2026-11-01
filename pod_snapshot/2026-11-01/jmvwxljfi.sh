cd /workspace/_r5_stage
python3 - <<EOF
import json
a=json.loads(open("/workspace/a3_eval/demos_wrongclosure_v9/adjudication_wc.json").read())
print(json.dumps(a["tally"]))
for e in a["episodes"]:
    rc = e["recovery_clamp"] or {}
    print(e["seed"], e["cut"], "abn_peak", e["wrong_closure_peak_pad_n"], "at_grip", e["wrong_closure_max_at_grip"], "strict", e["strict_accept"], "closure", rc.get("closure_peak_pad_n"), "carry", rc.get("carry_peak_pad_n"), "grip", rc.get("max_cookies_at_grip"), "lift", rc.get("carried_lift_mm"), "up", rc.get("placed_upright"), "idx", rc.get("carried_index"))
r=json.loads(open("/workspace/a3_eval/demos_wrongclosure_v9/summary.json").read())
print([ (int(x["seed"]), x.get("extras_released_in_correction"), x["stop"]) for x in r ])
EOF