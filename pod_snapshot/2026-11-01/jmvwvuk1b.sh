ls -l /workspace/a3_eval/r5_b0_c0_paired/ | tail -20
echo PS
ps -eo pid,etime,cmd --sort=start_time | grep -E "run_matrix|autonomy_metrics|replay_wrongclosure|collect" | grep -v grep
echo R5M
tail -c 200 /workspace/r5metrics.out 2>&1
echo DF
df -h /workspace | tail -1