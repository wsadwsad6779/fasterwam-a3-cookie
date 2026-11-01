ls /workspace/a3_data/a3_fw
ls /workspace/a3_data/a3_fw/meta 2>/dev/null
echo ==INFO==
head -c 1200 /workspace/a3_data/a3_fw/meta/info.json 2>/dev/null
echo
echo ==EPISODES_HEAD==
head -c 1200 /workspace/a3_data/a3_fw/meta/episodes.jsonl 2>/dev/null
echo
echo ==SUMMARY_SEARCH==
find /workspace -maxdepth 5 -name collection_summary.json 2>/dev/null | head
find /workspace -maxdepth 5 -name tasks.jsonl 2>/dev/null | head
echo ==TASKS==
for f in $(find /workspace -maxdepth 5 -name tasks.jsonl 2>/dev/null | head -3); do echo "-- $f"; head -3 $f; done
echo DONE