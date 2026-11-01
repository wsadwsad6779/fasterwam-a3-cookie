pgrep -af 'train_light' | head -5
echo '--- exec log ---'
cat /workspace/_r4_stage/dry.exec.log | tail -n 25
echo '--- verify.log ---'
ls -l /workspace/a3_train_runs/_r5_dryrun/
