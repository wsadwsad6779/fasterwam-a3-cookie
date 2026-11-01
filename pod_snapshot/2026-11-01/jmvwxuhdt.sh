ls -d /workspace/a3_eval/demos_*
python3 - <<EOF
import json
r=json.loads(open("/workspace/a3_eval/demos_single_v30b/summary.json").read())
print("rows", len(r), "accept_true", sum(1 for x in r if x.get("accept")))
print("clamp_keys", sorted((r[0].get("clamps") or [{}])[-1].keys()))
EOF