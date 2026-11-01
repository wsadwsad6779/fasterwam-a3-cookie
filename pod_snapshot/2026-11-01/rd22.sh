cd /workspace
ls /workspace/data/ 2>/dev/null | head
D=/workspace/data/a3_front_close_left_100
ls $D
find $D -maxdepth 2 -name 'info.json' | head
python3 - <<'PY'
import json, glob
for p in glob.glob('/workspace/data/a3_front_close_left_100/**/info.json', recursive=True)[:2]:
    print(p)
    d = json.load(open(p))
    feats = d.get('features', {})
    for k, v in feats.items():
        print('  ', k, v.get('dtype'), v.get('shape'))
PY
ls /workspace/a3_data/a3_fw/
python3 - <<'PY'
import json
p='/workspace/a3_data/a3_fw/summary.json'
d = json.load(open(p))
print(json.dumps(d, indent=1)[:1200])
PY
tail -c 300 /workspace/gate10.out
