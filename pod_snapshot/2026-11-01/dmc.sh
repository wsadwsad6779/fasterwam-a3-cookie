cd /workspace/_r5_stage
python3 -m py_compile count_accepts.py && echo COMPILE_OK
python3 count_accepts.py /workspace/a3_eval/demos_single_v30 /workspace/a3_eval/demos_single_v30b 2>&1 | head -80
echo __END__