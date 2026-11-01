echo '--- dry tail ---'
tail -c 1200 /workspace/_r4_stage/dry.exec.log 2>/dev/null
echo '--- train.log tail ---'
tail -n 12 /workspace/a3_train_runs/_r5_dryrun/train.log 2>/dev/null
echo '--- gen scripts ---'
grep -rln --include=*.py -e 'ep_start' -e 'a3_fw' /workspace/FasterWAM/scripts /workspace/FasterWAM/src 2>/dev/null | head -20
echo '--- expert collectors ---'
ls /workspace/FasterWAM/scripts | head -60
