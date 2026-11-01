ls -1 /workspace/a3_eval/r5_b0_c0_paired/*.npz | wc -l
ls -1 /workspace/*.out | tail -12
echo C1TALLY
grep -h "loose" /workspace/cc*.out /workspace/c1*.out 2>/dev/null | tail -8
echo WATCH
tail -c 200 /workspace/r5metrics.out
echo WC5BLAUNCHED
sleep 3; head -c 300 /workspace/wc5b.out