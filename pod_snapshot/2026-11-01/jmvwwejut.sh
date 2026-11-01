sha256sum /workspace/_r4_stage/autonomy_metrics.py /workspace/_r5_stage/autonomy_metrics.py 2>&1 | head -4
grep -n autonomy /workspace/r5metrics.sh 2>/dev/null | head -8
echo R4DIR
ls -l --time-style=+%m-%d_%H:%M /workspace/a3_eval/r4_b0_c0_paired | tail -6
echo WC5B
tail -c 900 /workspace/wc5b.out