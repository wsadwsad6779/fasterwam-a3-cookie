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
    3000) SEEDS=7100,7101,7102,7103,7104,7105,7106,7107,7108,7109 ;;
    6000) SEEDS=7200,7201,7202,7203,7204,7205,7206,7207,7208,7209 ;;
    9000) SEEDS=7300,7301,7302,7303,7304,7305,7306,7307,7308,7309 ;;
    12000) SEEDS=7400,7401,7402,7403,7404,7405,7406,7407,7408,7409 ;;
    15000) SEEDS=7500,7501,7502,7503,7504,7505,7506,7507,7508,7509 ;;
    18000) SEEDS=7600,7601,7602,7603,7604,7605,7606,7607,7608,7609 ;;
    21000) SEEDS=7700,7701,7702,7703,7704,7705,7706,7707,7708,7709 ;;
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