cd /workspace
tail -n 1 a3_train_runs/a3_lora_r5/train.log
pgrep -af replay_forces.py | head -3
ls a3_eval | grep -e demos -e 10clamp
ls -la a3_eval/replay_single_10clamp | head -5
tail -c 700 /workspace/variants.out
df -BG /workspace | tail -1
