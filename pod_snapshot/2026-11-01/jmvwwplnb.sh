grep -o ["]observation[a-z_.]*["] /workspace/a3_data/a3-front-close-left-100/meta/info.json | sort -u
echo ===
grep -n shape /workspace/a3_data/a3-front-close-left-100/meta/info.json | head -5
echo ===
python3 - <<PY
import json
d=json.load(open("/workspace/a3_data/a3-front-close-left-100/meta/info.json"))
print(d.get("total_episodes"), d.get("total_frames"))
for k,v in d["features"].items(): print(k, v.get("dtype"), v.get("shape"))
PY