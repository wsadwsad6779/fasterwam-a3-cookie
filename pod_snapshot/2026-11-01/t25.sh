cd /workspace
python3 - <<PY
import json
rows = json.load(open("/workspace/a3_eval/pilot_single_lowforce_noisy/summary.json"))
for r in rows:
    print(r["seed"], r["steps"], r["stop"], "in_target", r["in_target_final"], "padpeak", round(r["pad_peak_this_run"], 3))
    for b in r["batch_reports"]:
        print("   ", b.get("cookies"), "lifted", b.get("lifted"), "released", b.get("released"), "h", [round(x, 4) for x in b.get("lift_heights_m", [])])
PY
echo DONE