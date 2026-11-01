#!/bin/bash
echo "== mask job =="
tail -4 /workspace/_r4_stage/tko_b0.log
echo "== r4 run dir =="
ls -la /workspace/a3_train_runs/a3_lora_r4 | head -25
echo "== r4 checkpoints =="
ls -la /workspace/a3_train_runs/a3_lora_r4/checkpoints 2>&1 | head -12
ls -la /workspace/a3_train_runs/a3_lora_r4/checkpoints/light 2>&1 | head -12
ls -la /workspace/a3_train_runs/a3_lora_r4/checkpoints/light/step_004000 2>&1 | head -20
ls -la /workspace/a3_train_runs/a3_lora_r4/checkpoints/full 2>&1 | head -12
ls -la /workspace/a3_train_runs/a3_lora_r4/checkpoints/step_004000 2>&1 | head -20
echo "== yaml/json in run dir =="
ls -la /workspace/a3_train_runs/a3_lora_r4/*.yaml /workspace/a3_train_runs/a3_lora_r4/*.json /workspace/a3_train_runs/a3_lora_r4/*.log 2>&1 | head -20
echo "== r3 run dir =="
ls -la /workspace/a3_train_runs/a3_lora_r3 | head -20
ls -la /workspace/a3_train_runs/a3_lora_r3/checkpoints/light 2>&1 | head -12
echo "== train runs top =="
ls -la /workspace/a3_train_runs | head -20
echo "== workspace top =="
ls /workspace | head -40
echo "== disk =="
df -h /workspace | tail -1
echo "== DONE =="