ls -1 /workspace/a3_train_runs/a3_lora_r4/checkpoints/light 2>/dev/null
echo R5CKPT
ls -1 /workspace/a3_train_runs/a3_lora_r5/checkpoints/light 2>/dev/null
echo AEVAL2
ls -1t /workspace/a3_eval | grep -iE "r5|r4_b0|paired" | head -12
echo METRICS
head -30 /workspace/a3_eval/r5_autonomy_metrics.txt