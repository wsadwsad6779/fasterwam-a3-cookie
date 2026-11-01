cd /workspace/_r5_stage
PYTHONPATH=/workspace/a3_dual_arm_sim/src python3 adjudicate_c3.py --dirs /workspace/a3_eval/demos_correct_c6 --write 2>&1 | tail -40