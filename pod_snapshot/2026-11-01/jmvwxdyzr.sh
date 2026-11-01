cd /workspace/_r5_stage
export PYTHONPATH=/workspace/a3_dual_arm_sim/src
wc -c adjudicate_c3.py
sha256sum adjudicate_c3.py
python3 -m py_compile adjudicate_c3.py && echo COMPILE_OK
python3 adjudicate_c3.py --dirs /workspace/a3_eval/demos_correct_c2 /workspace/a3_eval/demos_correct_c5 2>&1 | tail -20