grep -m3 -nE "max_steps|resume" /workspace/a3_train_runs/a3_lora_r4/config.yaml
echo R5RESTART
cat /workspace/a3_train_runs/a3_lora_r5/lr_restart.json
echo R4METRICS
head -16 /workspace/a3_eval/r4_autonomy_metrics_rerun.txt