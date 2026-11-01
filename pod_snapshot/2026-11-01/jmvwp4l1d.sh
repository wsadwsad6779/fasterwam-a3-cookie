head -c 900 /workspace/data/a3_front_close_left_100/attempts.jsonl
echo
echo '--- keys ---'
python -c "import json,sys;rows=[json.loads(l) for l in open('/workspace/data/a3_front_close_left_100/attempts.jsonl')];print(len(rows));print(sorted(rows[0].keys()));import collections;c=collections.Counter();
for r in rows:
  c[tuple(sorted(r.keys()))]+=1
print(c.most_common(3))"
echo '--- summary ---'
head -c 1200 /workspace/data/a3_front_close_left_100/collection_summary.json
