#!/usr/bin/env bash
# Hold until the r5 training process exits, then run the paired evaluation once.
W=/workspace/_r5_stage/r5_watch.log
echo "watch start $(date -u +%FT%TZ)" > "$W"
while pgrep -f train_light_r5.py > /dev/null; do sleep 60; done
echo "training exited $(date -u +%FT%TZ)" >> "$W"
tail -4 /workspace/a3_train_runs/a3_lora_r5/train.log >> "$W"
bash /workspace/_r5_stage/r5_eval.sh
echo "r5_eval.sh exit=$? $(date -u +%FT%TZ)" >> "$W"
