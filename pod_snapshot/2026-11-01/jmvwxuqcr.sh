python3 - <<EOF
import json
r=json.loads(open("/workspace/a3_eval/demos_single_v30b/summary.json").read())
print("rowkeys", sorted(r[0].keys()))
print("row_accept", sum(1 for x in r if x.get("accept")))
print("clamp_accept", sum(1 for x in r if (x.get("clamps") or [{}])[-1].get("accept")))
print("success", sum(1 for x in r if x.get("success")))
print("n_clamps", sorted(len(x.get("clamps") or []) for x in r))
EOF