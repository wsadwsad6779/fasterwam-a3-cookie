cd /workspace/a3_train_runs
du -sh a3_lora_r1/* a3_lora_r2/* 2>/dev/null | sort -h | tail -6
echo R3
du -sh a3_lora_r3/* 2>/dev/null | sort -h | tail -8
echo FREEMORE
du -sh /workspace/a3_eval/demos_single_v30b /workspace/a3_data/a3_fw 2>/dev/null
df -B1 /workspace | tail -1