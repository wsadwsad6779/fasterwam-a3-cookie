cd /workspace/a3_eval
python3 /workspace/_r5_stage/c3_summary.py demos_correct_c1 demos_correct_c2 demos_correct_c3 demos_correct_c4 demos_correct_c5 demos_correct_c6
echo C7NPZ
ls /workspace/a3_eval/demos_correct_c7/*.npz | wc -l
ls -d /proc/478449 2>/dev/null