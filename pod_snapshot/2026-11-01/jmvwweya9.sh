grep -c extras_released_in_correction.: true /workspace/wc5b.out
echo WC4STATUS
ps -eo pid,etime,cmd | grep -c replay_wrongclosure
grep -E "TALLY|^\{" /workspace/wc4.out | tail -8
echo V6B
ls -l /workspace/a3_eval/demos_wrongclosure_v6b 2>&1 | tail -8