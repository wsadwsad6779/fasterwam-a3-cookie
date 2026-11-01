find /workspace/a3_train_runs/a3_lora_r5 -maxdepth 3 | sort | head -40
echo ---
ls -la /workspace/a3_train_runs/a3_lora_r5/
echo --- 
find /workspace -maxdepth 4 -name "env.py" 2>/dev/null | head -10