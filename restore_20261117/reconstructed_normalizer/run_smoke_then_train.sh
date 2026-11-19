#!/bin/bash
cd /workspace/eval_restore_20261117/reconstructed_normalizer || exit 1
DATA=/workspace/eval_restore_20261117/reconstructed_normalizer/teacher_replay_20261118
rm -rf model_repair_runs/smoke5 model_repair_runs/lora_300
TRAIN_SMOKE=1 TRAIN_DATA_DIR=$DATA TRAIN_OUT=model_repair_runs/smoke5 \
  /opt/venv/bin/python train_lora_300.py > train_smoke5.log 2>&1
if grep -q TRAIN_DONE train_smoke5.log; then
  TRAIN_STEPS=300 TRAIN_DATA_DIR=$DATA TRAIN_OUT=model_repair_runs/lora_300 \
    /opt/venv/bin/python train_lora_300.py > train_lora_300.log 2>&1
  echo "TRAIN_EXIT=$?" >> train_lora_300.log
else
  echo "SMOKE_FAILED" > train_lora_300.log
fi
