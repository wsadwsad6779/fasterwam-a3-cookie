echo '--- lr_restart.json ---'
cat /workspace/a3_train_runs/_r5_dryrun/lr_restart.json
echo '--- r4 dir untouched ---'
ls -l --time-style=+%F_%T /workspace/a3_train_runs/a3_lora_r4/checkpoints/light/step_004000/
sha256sum /workspace/a3_train_runs/a3_lora_r4/checkpoints/light/step_004000/*.pt /workspace/a3_train_runs/a3_lora_r4/checkpoints/light/step_004000/trainer_state.json
