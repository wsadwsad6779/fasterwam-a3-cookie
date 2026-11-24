#!/bin/bash
# Watchdog for the approved V3 run. Restarts only when neither training, eval
# nor the milestone chain is running and the next milestone file is missing.
LOG=/workspace/fwv1/logs/t7_watchdog_v3run.log
RUN=/workspace/fwv1/train_runs/t7_21k_v3run
CONSOLE=/workspace/fwv1/logs/t7_train_v3run_console.log
TRAIN=/workspace/_t7_train.py
CHAIN=/workspace/_t7_chain_v3run.sh
export T7_RUN_DIR="$RUN"
export T7_LOG=/workspace/fwv1/logs/t7_train_v3run.log
export T7_STATE=/workspace/fwv1/p_items/POD_JOB_STATE.json
export T7_FORCE_DROPOUT=0.1
export T7_AUX_FORCE_W=0.05
exec >> "$LOG" 2>&1
echo "[watchdog] start $(date -u +%Y-%m-%dT%H:%M:%SZ)"
while true; do
  if [ -f /workspace/fwv1/STOP_TRAINING ]; then
    echo "[watchdog] STOP flag present; exiting $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    exit 0
  fi
  if ! pgrep -f "_t7_train.py" > /dev/null && ! pgrep -f "_t7_eval.py" > /dev/null && ! pgrep -f "_t7_chain_v3run.sh" > /dev/null; then
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
        if ! pgrep -f "_t7_chain_v3run.sh" > /dev/null; then
          echo "[watchdog] relaunch milestone chain $(date -u +%Y-%m-%dT%H:%M:%SZ)"
          nohup bash "$CHAIN" >> /workspace/fwv1/logs/t7_chain_v3run_console.log 2>&1 &
        fi
        sleep 20
      fi
    fi
  fi
  sleep 30
done