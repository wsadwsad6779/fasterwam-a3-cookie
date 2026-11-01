ls /workspace/a3_data/ | head -20
find /workspace/a3_data -maxdepth 3 -name collection_summary.json | head -10
echo ==META==
for f in $(find /workspace/a3_data -maxdepth 3 -name collection_summary.json | head -3); do echo "-- $f"; head -c 1800 $f; echo; done
echo DONE