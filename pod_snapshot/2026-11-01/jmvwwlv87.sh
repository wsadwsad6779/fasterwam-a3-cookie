cd /workspace/_r5_stage
python3 -m py_compile adjudicate_correct.py && echo COMPILE_OK
sha256sum adjudicate_correct.py
PYTHONPATH=/workspace/a3_dual_arm_sim/src python3 adjudicate_correct.py --dirs /workspace/a3_eval/demos_correct_c3 --write 2>&1 | tail -6