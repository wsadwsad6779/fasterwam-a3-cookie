date -u
for t in wc4 wc6fix cc4 cc5; do echo == $t; tail -c 150 /workspace/$t.out; done
echo PS
ps -eo pid,etime,cmd | grep -E "replay_correct|replay_wrongclosure" | grep -v grep | wc -l
df -B1 /workspace | tail -1