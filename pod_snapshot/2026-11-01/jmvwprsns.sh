nproc
uptime
ls /workspace/a3_train_runs/a3_lora_r5/checkpoints/light/
du -sh /workspace/a3_train_runs/a3_lora_r5
df -h /workspace | tail -1
grep -n "terminat\|truncated\|_fail\|drop" /workspace/FasterWAM/../a3_dual_arm_sim/sim/env.py 2>/dev/null | head -30
find /workspace -maxdepth 3 -name env.py -path "*sim*" 2>/dev/null | head