for t in wc4 wc6fix cc4 cc5; do printf "%s " $t; grep -c ^{ /workspace/$t.out; done
echo WC4TAIL
grep -E "TALLY|adjudication|WROTE" /workspace/wc4.out | tail -4
echo PROC
ps -eo pid,etime,cmd | grep -E "replay_correct|replay_wrongclosure" | grep -v grep | sed -e "s/ --.*out//"