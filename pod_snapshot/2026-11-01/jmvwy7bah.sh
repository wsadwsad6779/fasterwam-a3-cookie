cd /workspace/a3_eval
python3 /workspace/_r5_stage/c3_rows.py demos_correct_c6
echo RAWKEYS
python3 -c "import json;print(sorted(json.load(open(0))['episodes'][0].keys()))" < demos_correct_c6/adjudication_c3.json