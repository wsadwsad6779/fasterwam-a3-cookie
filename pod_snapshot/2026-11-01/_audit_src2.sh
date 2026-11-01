cd /workspace/FasterWAM || exit 1
{
echo "=== shuffle ==="
grep -rn "shuffle" src/fasterwam/runtime.py src/fasterwam/trainer.py | head -20
echo "=== batch/accum ==="
grep -rn -E "gradient_accumulation|batch_size|accum" src/fasterwam/trainer.py src/fasterwam/runtime.py | head -30
echo "=== r1 log resume msg ==="
sed -n "78,92p" /workspace/fasterwam-a3-cookie/finetune/logs/a3_lora_r1.log
echo "=== run_r3.sh ==="
cat /workspace/run_r3.sh
echo DONE
} > /workspace/_audit_src2.txt 2>&1
