#!/bin/bash
# Auto-restart the training segment if the process disappears before the current
# milestone checkpoint exists. Coexists with the milestone chain (it only acts
# when no _t7_train.py is running and the next milestone file is missing).
LOG=/workspace/fwv1/logs/t7_watchdog.log
RUN=/workspace/fwv1/train_runs/t7_21k
CONSOLE=/workspace/fwv1/logs/t7_train_console.log
TRAIN=/workspace/_t7_train.py
exec >> "$LOG" 2>&1
echo "[watchdog] start $(date -u +%Y-%m-%dT%H:%M:%SZ)"
while true; do
  if ! pgrep -f "_t7_train.py" > /dev/null && ! pgrep -f "_t7_eval.py" > /dev/null && ! pgrep -f "_t7_chain.sh" > /dev/null; then
    LARGEST=0
    for S in 3000 6000 9000 12000 15000 18000 21000; do
      F="$RUN/step_$(printf '%06d' $S).pt"
      if [ -f "$F" ]; then LARGEST=$S; fi
    done
    if [ "$LARGEST" -lt 21000 ]; then
      TARGET=$((LARGEST + 3000))
      if [ "$TARGET" -lt 3000 ]; then TARGET=3000; fi
      TARGET_FILE="$RUN/step_$(printf '%06d' $TARGET).pt"
      if [ ! -f "$TARGET_FILE" ]; then
        if [ -f "$RUN/latest.pt" ]; then
          echo "[watchdog] $(date -u +%H:%M:%SZ) restart from latest -> $TARGET"
          T7_RESUME="$RUN/latest.pt" T7_END_STEP=$TARGET T7_CKPT_EVERY=200 \
            nohup /opt/venv/bin/python "$TRAIN" >> "$CONSOLE" 2>&1 &
        elif [ "$LARGEST" -gt 0 ]; then
          SRC="$RUN/step_$(printf '%06d' $LARGEST).pt"
          echo "[watchdog] $(date -u +%H:%M:%SZ) restart from $SRC -> $TARGET"
          T7_RESUME="$SRC" T7_END_STEP=$TARGET T7_CKPT_EVERY=200 \
            nohup /opt/venv/bin/python "$TRAIN" >> "$CONSOLE" 2>&1 &
        else
          LAST=$(sed -n 's/.*"step": \([0-9]*\).*/\1/p' "$RUN/metrics.jsonl" 2>/dev/null | tail -1)
          if [ -z "$LAST" ]; then LAST=0; fi
          echo "[watchdog] $(date -u +%H:%M:%SZ) cold restart from step counter $LAST -> $TARGET"
          T7_START_STEP=$LAST T7_END_STEP=$TARGET T7_CKPT_EVERY=200 \
            nohup /opt/venv/bin/python "$TRAIN" >> "$CONSOLE" 2>&1 &
        fi
        if ! pgrep -f "_t7_chain.sh" > /dev/null; then
          echo "[watchdog] relaunch milestone chain $(date -u +%H:%M:%SZ)"
          nohup bash /workspace/_t7_chain.sh >> /workspace/fwv1/logs/t7_chain_console.log 2>&1 &
        fi
        sleep 20
      fi
    fi
  fi
  sleep 30
done