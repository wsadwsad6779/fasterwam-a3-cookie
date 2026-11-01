cd /workspace
L=/workspace/a3_train_runs/a3_lora_r5/train.log
ls -la $L
echo '=== r5 last ==='
tail -n 4 $L
echo '=== r5 checkpoints ==='
ls /workspace/a3_train_runs/a3_lora_r5/checkpoints/light
echo '=== lr_restart ==='
cat /workspace/a3_train_runs/a3_lora_r5/lr_restart.json 2>/dev/null | head -20
echo '=== full sweep2 ==='
cat /workspace/sweep2.out
echo '=== r4 ckpt hash ==='
sha256sum /workspace/a3_train_runs/a3_lora_r4/checkpoints/light/step_004000/adapter.pt
df -h /workspace | tail -1
