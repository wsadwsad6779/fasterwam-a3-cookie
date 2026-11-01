cd /workspace/_r5_stage
export PYTHONPATH=/workspace/a3_dual_arm_sim/src
while test -d /proc/467627 -o -d /proc/467763 -o -d /proc/473433; do sleep 30; done
echo ALLDONE $(date -u +%H:%M:%S)
df -B1 /workspace | tail -1
for f in /workspace/wc6fix.out /workspace/cc4.out /workspace/cc5.out; do echo == FILE $f; grep TALLY $f | tail -3; grep free_bytes $f | tail -2; done
echo == V9ADJ
python3 adjudicate_wc.py /workspace/a3_eval/demos_wrongclosure_v9 --write 2>&1 | tail -20
echo == C4C5ADJ
python3 adjudicate_correct.py --dirs /workspace/a3_eval/demos_correct_c4 /workspace/a3_eval/demos_correct_c5 --write 2>&1 | tail -40
echo == LS
ls -la /workspace/a3_eval/demos_wrongclosure_v9 /workspace/a3_eval/demos_correct_c4 /workspace/a3_eval/demos_correct_c5 | tail -40