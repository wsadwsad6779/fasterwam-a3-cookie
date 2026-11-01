#!/usr/bin/env bash
LOG=/workspace/r3.log
CK=/workspace/a3_train_runs/a3_lora_r3/checkpoints/weights/step_003000.pt
MERGED=/workspace/a3_train_runs/a3_lora_r3/step_003000_merged.pt
OLD=/workspace/a3_train_runs/a3_lora_r2/checkpoints/weights/step_003000.pt
echo "WATCH_START $(date +%H:%M:%S)" >> $LOG
N=0
while pgrep -f 'scripts/train.py' > /dev/null; do
  N=$((N+1))
  if [ "$N" -gt 2160 ]; then echo R3_TRAIN_FAIL watchdog_timeout >> $LOG; exit 4; fi
  sleep 20
done
echo "TRAIN_EXIT $(date +%H:%M:%S)" >> $LOG
if [ ! -s "$CK" ]; then echo R3_TRAIN_FAIL no_ckpt >> $LOG; exit 1; fi
CKSZ=$(stat -c %s "$CK")
if [ "$CKSZ" -lt 10000000000 ]; then echo "R3_TRAIN_FAIL small_ckpt $CKSZ" >> $LOG; exit 1; fi
echo "TRAIN_CKPT size=$CKSZ" >> $LOG
rm -f "$OLD"
FREE=$(df -P /workspace | awk 'NR==2{print $4}')
if [ "$FREE" -lt 13000000 ]; then echo "R3_MERGE_FAIL low_disk_kb=$FREE" >> $LOG; exit 2; fi
source /workspace/activate_fasterwam_a3.sh
cd /workspace/FasterWAM
python3 scripts/export_lora_merged.py data=a3_2cam task=a3_fasterwam +lora_ckpt=$CK +output=$MERGED >> $LOG 2>&1 || echo "R3_MERGE_FAIL export_rc=$?" >> $LOG
if [ ! -s "$MERGED" ]; then echo R3_MERGE_FAIL no_output >> $LOG; exit 3; fi
MSZ=$(stat -c %s "$MERGED")
if [ "$MSZ" -lt 10000000000 ]; then echo "R3_MERGE_FAIL small_output $MSZ" >> $LOG; exit 3; fi
echo "MERGE_OK size=$MSZ" >> $LOG
df -h /workspace | tail -1 >> $LOG
echo R3_DONE >> $LOG
