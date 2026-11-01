cd /workspace/a3_eval
python3 -c "import json;d=json.load(open(0));print(type(d).__name__, len(d));print(sorted(d[0].keys()) if isinstance(d,list) else sorted(d.keys()))" < demos_correct_c6/summary.json
echo WC_KEYS
python3 -c "import json;d=json.load(open(0));print(sorted(d['episodes'][0].keys()))" < demos_wrongclosure_v9/adjudication_wc.json