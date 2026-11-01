nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
df -h /workspace | tail -1
ls /workspace/a3_train_runs/a3_lora_r5/checkpoints/light/ 2>/dev/null
tail -c 1200 /workspace/a3_train_runs/a3_lora_r5/train.log
