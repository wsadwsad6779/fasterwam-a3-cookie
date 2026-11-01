sed -n '25,60p' /workspace/fasterwam_a3_bridge/bridge/_next/env_factory.py
echo ==R5==
ls /workspace/a3_train_runs/a3_lora_r5/checkpoints/light 2>/dev/null | tail -5
tail -c 400 /workspace/a3_train_runs/a3_lora_r5/train.log
df -h /workspace | tail -1
echo DONE