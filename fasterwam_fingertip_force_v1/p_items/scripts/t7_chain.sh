#!/bin/bash
# Milestone chain: wait for the running segment, then for each 3k milestone:
# verify checkpoint -> 10-seed eval + dual-view videos -> resume next segment.
LOG=/workspace/fwv1/logs/t7_chain.log
RUN=/workspace/fwv1/train_runs/t7_21k
EVAL=/workspace/_t7_eval.py
TRAIN=/workspace/_t7_train.py
export HF_ENDPOINT=https://hf-mirror.com
mkdir -p /workspace/fwv1/logs
exec >> "$LOG" 2>&1
echo "[chain] start $(date -u +%Y-%m-%dT%H:%M:%SZ); waiting for current segment"
while pgrep -f "_t7_train.py" > /dev/null; do
  sleep 30
done
echo "[chain] current segment finished $(date -u +%Y-%m-%dT%H:%M:%SZ)"
if [ -f "$RUN/latest.pt" ] && [ -f "$RUN/step_003000.pt" ]; then
  rm -f "$RUN/latest.pt"
  echo "[chain] removed duplicate latest.pt after step_003000 exists"
fi
for STEP in 3000 6000 9000 12000 15000 18000 21000; do
  SIX=$(printf "%06d" $STEP)
  CKPT="$RUN/step_${SIX}.pt"
  if [ ! -f "$CKPT" ]; then
    echo "[chain] MISSING $CKPT; stopping"
    exit 1
  fi
  case $STEP in
    3000) SEEDS=8100,8101,8102,8103,8104,8105,8106,8107,8108,8109 ;;
    6000) SEEDS=8200,8201,8202,8203,8204,8205,8206,8207,8208,8209 ;;
    9000) SEEDS=8300,8301,8302,8303,8304,8305,8306,8307,8308,8309 ;;
    12000) SEEDS=8400,8401,8402,8403,8404,8405,8406,8407,8408,8409 ;;
    15000) SEEDS=8500,8501,8502,8503,8504,8505,8506,8507,8508,8509 ;;
    18000) SEEDS=8600,8601,8602,8603,8604,8605,8606,8607,8608,8609 ;;
    21000) SEEDS=8700,8701,8702,8703,8704,8705,8706,8707,8708,8709 ;;
  esac
  echo "[chain] eval step=$STEP seeds=$SEEDS $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  T7_CKPT="$CKPT" T7_STEP=$STEP T7_SEEDS=$SEEDS T7_MAX_STEPS=600 T7_DENOISE=5 \
    /opt/venv/bin/python "$EVAL" >> "/workspace/fwv1/logs/eval_step_${STEP}.log" 2>&1
  RC=$?
  echo "[chain] eval rc=$RC step=$STEP"
  if [ $RC -ne 0 ]; then
    echo "[chain] eval failed; stopping for diagnosis"
    exit 1
  fi
  NEXT=$((STEP + 3000))
  if [ $NEXT -gt 21000 ]; then
    echo "[chain] all milestones done $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    break
  fi
  echo "[chain] resume train $STEP->$NEXT $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  T7_RESUME="$CKPT" T7_END_STEP=$NEXT T7_CKPT_EVERY=500 \
    /opt/venv/bin/python "$TRAIN" >> "/workspace/fwv1/logs/train_seg_${NEXT}.log" 2>&1
  TRC=$?
  echo "[chain] train rc=$TRC next=$NEXT"
  if [ $TRC -ne 0 ]; then
    echo "[chain] train failed; stopping for diagnosis"
    exit 1
  fi
  if [ -f "$RUN/latest.pt" ]; then
    rm -f "$RUN/latest.pt"
    echo "[chain] removed duplicate latest.pt after step_$(printf "%06d" $NEXT).pt"
  fi
done
echo "[chain] DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"