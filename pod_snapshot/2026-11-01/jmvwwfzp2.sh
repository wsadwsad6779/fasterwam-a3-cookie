head -6 /workspace/wc3.out
echo V6DIR
ls -1 /workspace/a3_eval/demos_wrongclosure_v6 | head -8
echo NPID
ps -eo pid,etime,cmd | grep -c replay_wrongclosure