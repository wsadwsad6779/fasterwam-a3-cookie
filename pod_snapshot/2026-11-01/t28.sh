tail -c 1500 /workspace/sweep.out
echo ==JOBS==
pgrep -af pilot_sweep | head -3
echo ==R5==
ls /workspace/a3_train_runs/a3_lora_r5/checkpoints/light | tail -3
tail -c 260 /workspace/a3_train_runs/a3_lora_r5/train.log
df -h /workspace | tail -1
echo DONE