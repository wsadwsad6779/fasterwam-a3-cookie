cd /workspace/a3_eval
for d in demos_single_v30 demos_single_v30b; do echo "== $d"; ls -1 $d 2>/dev/null | wc -l; python3 -c "import json,sys;rows=json.load(open(sys.argv[1]));print(\"n\",len(rows),\"accept\",sum(1 for r in rows if r.get(\"accept\")),\"fail\",sum(1 for r in rows if not r.get(\"accept\")))" $d/summary.json 2>&1 | tail -2; done
tail -c 700 /workspace/d30b.out
echo __END__