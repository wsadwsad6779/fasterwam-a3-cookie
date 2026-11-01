ls -la /workspace/a3_eval/demos_correct_c7
echo NPZ
ls /workspace/a3_eval/demos_correct_c7/*.npz 2>/dev/null | wc -l
echo LOGTAIL
tail -12 /workspace/_r5_stage/c7_rec.out 2>/dev/null
tail -12 /workspace/c7_rec.out 2>/dev/null