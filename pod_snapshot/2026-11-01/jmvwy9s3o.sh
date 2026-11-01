cd /workspace/a3_eval
python3 /workspace/_r5_stage/c3_rows.py demos_correct_c7
echo MANIFEST
python3 /workspace/_r5_stage/build_corpus_manifest.py --correct demos_correct_c1 demos_correct_c2 demos_correct_c3 demos_correct_c4 demos_correct_c5 demos_correct_c6 demos_correct_c7 --wc demos_wrongclosure_v8 demos_wrongclosure_v9 --single demos_single_v30b 2>&1 | tail -30