rm -rf /workspace/a3_train_runs/_r5_dryrun
rm -rf /tmp/r5up
pgrep -af train_light | head -3
df -B1 --output=avail /workspace | tail -1
nohup bash -c 'bash /workspace/_r4_stage/run_r5.sh > /workspace/_r4_stage/r5.exec.log 2>&1; echo DONE_MARKER_EXIT_0 >> /workspace/_r4_stage/r5.exec.log' >/dev/null 2>&1 &
sleep 40
echo '--- exec log ---'
tail -n 30 /workspace/_r4_stage/r5.exec.log 2>/dev/null
echo '--- proc ---'
pgrep -af train_light | head -3
