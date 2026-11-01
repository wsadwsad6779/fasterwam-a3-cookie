cd /workspace/_r5_stage
sed -n 1,10p adjudicate_wc.py
echo == HEADER_CORRECT
sed -n 1,20p adjudicate_correct.py
echo == ROWKEYS
python3 - <<EOF
import json
r=json.loads(open("/workspace/a3_eval/demos_correct_c2/summary.json").read())[0]
print(sorted(r.keys()))
print(json.dumps(r.get("retry_log"),indent=0)[:900])
EOF