cd /workspace/_r5_stage
python3 - <<EOF
import json
for d in ["c1","c2","c3"]:
    rows=json.loads(open("/workspace/a3_eval/demos_correct_%s/summary.json"%d).read())
    for r in rows:
        for e in r["retry_log"]:
            if e["anomaly_kind"] in ("empty_grasp","stuck_no_force"):
                print(d, r["seed"], r["variant"], r["stop"], r["accept"], r["recovered_and_valid"], r["pad_max_n"], r["frames_over_2n"], len(r["clamps"]), json.dumps(e))
EOF