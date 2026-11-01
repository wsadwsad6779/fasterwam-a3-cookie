cd /workspace || exit 1
{
echo "=== workspace logs ==="
ls -la *.log 2>&1
echo "=== ckpt dirs ==="
du -sh a3_train_runs/a3_lora_r2/checkpoints a3_train_runs/a3_lora_r3/checkpoints 2>&1
find a3_train_runs/a3_lora_r2/checkpoints a3_train_runs/a3_lora_r3/checkpoints -maxdepth 2 -type f -exec ls -la {} \; 2>&1 | head -40
echo "=== a3_fw ==="
ls -la a3_data/a3_fw 2>&1 | head -30
echo "=== a3_data dirs ==="
du -sh a3_data/* 2>&1
echo "=== run logs search ==="
ls -la a3_train_runs/ 2>&1
find /workspace -maxdepth 2 -name "*r3*" -o -maxdepth 2 -name "*train*.log" 2>/dev/null | head -20
echo CMDDONE
} > /workspace/_a3.txt 2>&1
