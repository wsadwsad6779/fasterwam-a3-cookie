cd /workspace || exit 1
{
echo "=== RUN DIRS ==="
ls -la a3_train_runs/a3_lora_r1 a3_train_runs/a3_lora_r2 a3_train_runs/a3_lora_r3 2>&1
echo "=== R2 markers ==="
grep -c "Loading Wan" r2.log
grep -n "step=3000/3000" r2.log | head -5
grep -c "No space left" r2.log
echo "=== R2 first/last step lines ==="
grep -E "step=[0-9]+/3000" r2.log | head -3
grep -E "step=[0-9]+/3000" r2.log | tail -3
echo "=== R2 lr trace every 400 ==="
grep -oE "step=[0-9]+/3000" r2.log | awk -F= "{print $2}" | cut -d/ -f1 | awk "NR%20==1" | head -60
echo "=== R1 log path ==="
ls -la fasterwam-a3-cookie/finetune/logs/ 2>&1
echo CMDDONE
} > /workspace/_a2.txt 2>&1
