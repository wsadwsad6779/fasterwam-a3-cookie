cd /workspace/_r5_stage
export PYTHONPATH=/workspace/a3_dual_arm_sim/src
while test -d /proc/478449 -o -d /proc/478581; do sleep 30; done
echo ALLDONE $(date -u +%H:%M:%S)
echo == C3_6
python3 adjudicate_c3.py --dirs /workspace/a3_eval/demos_correct_c6 --write
echo == C3_7
python3 adjudicate_c3.py --dirs /workspace/a3_eval/demos_correct_c7 --write
echo == C1V2
python3 adjudicate_correct.py --dirs /workspace/a3_eval/demos_correct_c6 /workspace/a3_eval/demos_correct_c7 --write --suffix _v2 > /workspace/_r5_stage/c67_c1.out 2>&1
grep -E .dir. /workspace/_r5_stage/c67_c1.out
grep -E WROTE /workspace/_r5_stage/c67_c1.out
echo == DISK
df -B1 /workspace | tail -1