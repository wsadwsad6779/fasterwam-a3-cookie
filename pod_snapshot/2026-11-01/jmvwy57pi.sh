cd /workspace/a3_eval
python3 -c "import json;d=json.load(open('demos_correct_c6/adjudication_c3.json'));print(type(d));print(list(d.keys())[:12] if isinstance(d,dict) else d[:1])"