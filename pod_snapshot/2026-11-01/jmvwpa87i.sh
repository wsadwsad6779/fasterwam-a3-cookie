head -c 700 /workspace/data/a3_front_close_left_100/a3_episode_metadata.jsonl
echo
echo '--- summary tail ---'
python - <<'EOF'
import json
d=json.load(open('/workspace/data/a3_front_close_left_100/collection_summary.json'))
print(list(d.keys()))
for k,v in d.items():
    if k=='config': continue
    s=json.dumps(v)
    print(k, s[:400])
EOF
