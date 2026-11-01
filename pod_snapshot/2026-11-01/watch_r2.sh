#!/usr/bin/env bash
set -u
source /workspace/activate_fasterwam_a3.sh
LOG=/workspace/watch_r2.log
CKPT=/workspace/a3_train_runs/a3_lora_r2/checkpoints/weights/step_003000.pt
MERGED=/workspace/a3_train_runs/a3_lora_r2/step_003000_merged.pt
echo "$(date) watcher2 start" > $LOG
for i in $(seq 1 480); do
  if grep -q "max_steps reached" /workspace/r2.log; then break; fi
  if grep -q "Traceback" /workspace/r2.log && ! pgrep -f "scripts/train.py" > /dev/null; then echo "$(date) TRAIN_DIED" >> $LOG; exit 1; fi
  sleep 30
done
echo "$(date) train finished" >> $LOG
if [ ! -f "$CKPT" ]; then echo "$(date) CKPT_MISSING $CKPT" >> $LOG; ls -l /workspace/a3_train_runs/a3_lora_r2/checkpoints/weights >> $LOG; exit 1; fi
rm -f /workspace/a3_train_runs/a3_lora_r2/checkpoints/weights/step_001500.pt
echo "$(date) removed 1500 ckpt; free=$(df -h /workspace | tail -1 | awk '{print $4}')" >> $LOG
cd /workspace/FasterWAM
python3 scripts/export_lora_merged.py data=a3_2cam task=a3_fasterwam \
  lora_ckpt=$CKPT \
  output=$MERGED >> $LOG 2>&1
RC=$?
if [ $RC -eq 0 ] && [ -f "$MERGED" ]; then echo "$(date) MERGE_DONE $(stat -c%s $MERGED)" >> $LOG; else echo "$(date) MERGE_FAILED rc=$RC" >> $LOG; fi
df -h /workspace | tail -1 >> $LOG
