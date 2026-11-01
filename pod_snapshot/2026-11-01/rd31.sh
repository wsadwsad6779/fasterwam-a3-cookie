cd /workspace
python3 - <<PY
import json
s=json.load(open("/workspace/a3_eval/replay_single_gate10/summary.json"))
print(type(s), len(s))
print(json.dumps(s, indent=1)[:1500])
PY
