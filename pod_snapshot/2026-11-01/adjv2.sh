cd /workspace/_r5_stage
export PYTHONPATH=/workspace/a3_dual_arm_sim/src
python3 -m py_compile adjudicate_correct.py && echo COMPILE_OK
sha256sum adjudicate_correct.py
wc -c adjudicate_correct.py
python3 adjudicate_correct.py --dirs /workspace/a3_eval/demos_correct_c1 /workspace/a3_eval/demos_correct_c2 /workspace/a3_eval/demos_correct_c3 /workspace/a3_eval/demos_correct_c4 /workspace/a3_eval/demos_correct_c5 --write --suffix _v2
echo == TALLIES
grep -E .dir. /workspace/adjv2.out
echo == WROTE
grep -E WROTE /workspace/adjv2.out
echo == C3
python3 adjudicate_c3.py --dirs /workspace/a3_eval/demos_correct_c1 /workspace/a3_eval/demos_correct_c2 /workspace/a3_eval/demos_correct_c3 /workspace/a3_eval/demos_correct_c4 /workspace/a3_eval/demos_correct_c5 --write
echo == C67
ls /workspace/a3_eval/demos_correct_c6 | wc -l
ls /workspace/a3_eval/demos_correct_c7 | wc -l
df -B1 /workspace | tail -1