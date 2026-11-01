grep -m4 -nE "max_steps|initial_model|resume" /workspace/a3_train_runs/a3_lora_r5/config.yaml
echo R5EVAL
ls -1t /workspace/a3_train_runs/a3_lora_r5/eval | head -10
echo AEVALDIRS
ls -1t /workspace/a3_eval | head -14