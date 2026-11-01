echo "== d30b.sh"
cat /workspace/d30b.sh
echo "== gate10 launcher"
grep -nE "python|PYTHONPATH|source|conda|venv" /workspace/variants.sh /workspace/d30.sh 2>/dev/null | head -20
echo "== v30 keys"
python3 -c "import json;r=json.load(open('/workspace/a3_eval/demos_single_v30/summary.json'));print(sorted(r[0].keys()));print(len(r))"
echo __END__