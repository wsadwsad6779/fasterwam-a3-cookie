cd /workspace/_r5_stage
python3 -m py_compile check_correct.py && echo COMPILE_OK
export PYTHONPATH=/workspace/a3_dual_arm_sim/src
date
python3 check_correct.py /workspace/a3_eval/demos_correct_c1 2>&1 | tail -30
echo '--- cc1 lines ---'
grep -c . /workspace/cc1.out
echo __END__
