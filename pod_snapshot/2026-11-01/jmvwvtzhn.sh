date -u +%H:%M:%S
ls -1 /workspace/a3_eval/r5_b0_c0_paired/ | wc -l
ls -l /workspace/a3_eval/r5_autonomy_metrics.txt /workspace/a3_eval/r4_autonomy_metrics_rerun.txt 2>&1 | tail -3
grep -c METRICS_CHAIN_DONE /workspace/r5metrics.out 2>&1
echo CC3
tail -c 300 /workspace/cc3.out
echo WC4
tail -c 300 /workspace/wc4.out
echo WC5A
tail -c 500 /workspace/wc5a.out