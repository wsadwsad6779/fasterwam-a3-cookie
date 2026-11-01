cd /workspace/_r4_stage
rm -rf /workspace/a3_train_runs/_r5_dryrun
nohup bash -c 'bash /workspace/_r4_stage/run_r5_dry.sh > /workspace/_r4_stage/dry.exec.log 2>&1; echo \"__DRYDONE__\$?\" >> /workspace/_r4_stage/dry.exec.log' >/dev/null 2>&1 &
sleep 5
echo LAUNCHED
df -B1 --output=avail /workspace | tail -1
