cd /workspace
python3 -m py_compile _r5_stage/dump_summaries.py && echo COMPILE_OK
sha256sum _r5_stage/dump_summaries.py
python3 _r5_stage/dump_summaries.py a3_eval/r5_b0_c0_paired a3_eval/r4_b0_c0_paired 2>&1 | head -20