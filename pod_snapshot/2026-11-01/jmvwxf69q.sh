cd /workspace/_r5_stage
python3 - <<EOF
import json, os, glob
for d in sorted(glob.glob("/workspace/a3_eval/demos_correct_c*")):
    aj = d + "/adjudication_correct.json"
    if not os.path.exists(aj):
        print(d, "NO_ADJ"); continue
    adj = json.loads(open(aj).read())
    summ = json.loads(open(d + "/summary.json").read())
    kind = {int(r["seed"]): (r["retry_log"][0]["anomaly_kind"] if r["retry_log"] else "-") for r in summ}
    print("DIR", d, json.dumps(adj["tally"]))
    for e in adj["episodes"]:
        ab = [x["segments"] for x in e["segments"] if x["kind"] == "abnormal"][0]
        print("  ", e["seed"], kind.get(e["seed"], "?"), "strict", e["strict_accept"], "peak", ab["peak"], "o16", ab["over_1p6"], "o20", ab["over_2p0"], "rec_peak", [x["segments"] for x in e["segments"] if x["kind"] == "recovery"][0]["peak"])
EOF