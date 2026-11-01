cd /workspace/_r5_stage
export PYTHONPATH=/workspace/a3_dual_arm_sim/src
head -3 /workspace/a3_eval/demos_wrongclosure_v9/adjudication_wc.json
grep -o driver_sha[^,]* /workspace/a3_eval/demos_wrongclosure_v9/summary.json | head -2
ls /workspace/a3_eval/demos_wrongclosure_v6b/adjudication_wc.json 2>/dev/null || echo NO_V6B_ADJ
df -B1 /workspace | tail -1
python3 - <<EOF
import json, glob
for d in sorted(glob.glob("/workspace/a3_eval/demos_correct_c*")):
    rows=json.loads(open(d+"/summary.json").read())
    for r in rows:
        for e in r["retry_log"]:
            if e["anomaly_kind"] != "excessive_pad_force":
                print(d.split("_")[-1], r["seed"], r["variant"], r["accept"], r["recovered_and_valid"], r["pad_max_n"], r["frames_over_2n"], e["anomaly_kind"], e["peak_pad_n"], e["at_step"], r["steps"])
EOF