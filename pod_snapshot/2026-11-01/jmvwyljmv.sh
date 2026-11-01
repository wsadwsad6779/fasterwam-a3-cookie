cd /workspace/a3_train_runs
find . -name '*merged*.pt' -printf '%s %TY-%Tm-%Td_%TH:%TM %p\n'
echo RESUME_CHAIN
for r in a3_lora_r2 a3_lora_r3 a3_lora_r4 a3_lora_r5; do echo $r; grep -E "resume" $r/config.yaml; done
echo R1R2REFS
grep -rn 'a3_lora_r1/step_002000_merged\|a3_lora_r2/step_003000_merged' /workspace --include=*.py --include=*.sh --include=*.yaml --include=*.json 2>/dev/null | grep -v '.ipynb_checkpoints' | head -12