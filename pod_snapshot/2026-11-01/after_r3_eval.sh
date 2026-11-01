#!/usr/bin/env bash
LOG=/workspace/r3.log
MERGED=/workspace/a3_train_runs/a3_lora_r3/step_003000_merged.pt
OUTF=/workspace/r3_c0_metrics.txt
N=0
while true; do
  if grep -q 'R3_TRAIN_FAIL\|R3_MERGE_FAIL' $LOG; then
    echo "ABORT $(grep 'R3_TRAIN_FAIL\|R3_MERGE_FAIL' $LOG | tail -1)" > $OUTF
    exit 1
  fi
  if grep -q R3_DONE $LOG; then break; fi
  N=$((N+1))
  if [ "$N" -gt 720 ]; then echo ABORT watcher_timeout > $OUTF; exit 1; fi
  sleep 30
done
echo "start $(date +%H:%M:%S)" > $OUTF
echo "ckpt=$MERGED size=$(stat -c %s $MERGED)" >> $OUTF
cd /workspace/fasterwam_a3_bridge/bridge
rm -f outputs/r3/C0/*.npz outputs/r3/C0/*_front.mp4
bash /workspace/run_stage_r2.sh C0 "$MERGED" r3 > /workspace/r3c0.log 2>&1
RC=$?
echo "eval_rc=$RC" >> $OUTF
if ! grep -q "loaded_ckpt=$MERGED" /workspace/r3c0.log; then echo CKPT_MISMATCH >> $OUTF; fi
/opt/venv/bin/python /workspace/r2_metrics.py outputs/r3/C0 >> $OUTF 2>&1
echo "metrics_done $(date +%H:%M:%S)" >> $OUTF
echo ALL_DONE >> $OUTF
