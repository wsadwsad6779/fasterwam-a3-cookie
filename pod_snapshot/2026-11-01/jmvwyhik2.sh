date -u +%H:%M:%S
ls -1t /workspace/a3_train_runs/ | head -8
echo R5LOG
ls -1t /workspace/a3_train_runs/a3_lora_r5/ 2>/dev/null | head -12
echo GP
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader
echo PYS
ps -eo pid,etime,cmd | grep -c 'torchrun|train_a3'
df -BG /workspace | tail -1