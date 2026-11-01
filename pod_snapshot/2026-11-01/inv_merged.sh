cd /workspace/a3_train_runs
echo SIZES
du -sh a3_lora_r1 a3_lora_r2 a3_lora_r3 a3_lora_r4 a3_lora_r5 2>/dev/null
echo MERGED
find . -name '*merged*.pt' -printf '%s\t%TY-TM-TD\t%p\n' | sort -k3
echo BIG
find /workspace -maxdepth 3 -type f -size +2G -printf '%s\t%TY-TM-TD\t%p\n' | sort -rn | head -25
echo REFS
grep -rlE 'merged' /workspace/_r4_stage /workspace/_r5_stage /workspace/a3_train_runs/*/config.yaml 2>/dev/null | head -20
echo DONE