ls -la /workspace/sweep2.out /workspace/sweep.out
pgrep -af pilot_sweep2 | head -3
ls /workspace/a3_eval/ | grep pilot
echo ==E==
ls -la /workspace/a3_eval/pilot_single_E_rows_lift0012_f045/ 2>/dev/null
cat /workspace/sweep2.out
echo DONE