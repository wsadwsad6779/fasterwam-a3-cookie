#!/bin/bash
# Segmented action-expert adaptation driver: 500+500+1000+2000+4000+4000 = 12,000 steps.
# Trains on corrected teacher data, validates on the held-out episode, evaluates milestones.
set -u
cd /workspace/eval_restore_20261117/reconstructed_normalizer || exit 1
LOG=drive_segments.log
WORK=/root/repair_runs/action_expert_v2
PUB=/workspace/eval_restore_20261117/reconstructed_normalizer/model_repair_runs/action_expert_v2
EV=/workspace/eval_restore_20261117/reconstructed_normalizer/c_class_eval_20261119
ts() { date -u +%FT%TZ; }
echo "DRIVE_START $(ts)" >> "$LOG"

CUM=0
FIRST=1
for S in 500 500 1000 2000 4000 4000; do
  PREV=$CUM
  CUM=$((CUM + S))
  if [ $FIRST -eq 1 ]; then RESUME=0; else RESUME=1; fi
  FIRST=0
  SEG_STEPS=$S TRAIN_RESUME=$RESUME TRAIN_WORK=$WORK TRAIN_PUB=$PUB \
    /opt/venv/bin/python train_action_expert_segmented_20261119.py > "seg_${CUM}.log" 2>&1
  RC=$?
  echo "SEG to=$CUM rc=$RC $(ts)" >> "$LOG"
  if [ $RC -ne 0 ] || ! grep -q SEGMENT_DONE "seg_${CUM}.log"; then
    echo "SEGMENT_FAILED at $CUM $(ts)" >> "$LOG"
    exit 1
  fi
  VAL=$(tail -n 1 "$WORK/segments.jsonl" | /opt/venv/bin/python -c "import sys,json;print(json.load(sys.stdin)['val_loss'])")
  echo "VAL $CUM $VAL $(ts)" >> "$LOG"
  # stop when validation degrades clearly at two consecutive milestones
  /opt/venv/bin/python - "$CUM" "$VAL" <<'PY' >> "$LOG" 2>&1
import json, sys
cum, val = int(sys.argv[1]), float(sys.argv[2])
p='/workspace/eval_restore_20261117/reconstructed_normalizer/drive_val.json'
try:
    d=json.load(open(p))
except Exception:
    d={'history':[]}
d['history'].append([cum,val])
worst=d.get('best',1e9)
d['best']=min(worst,val)
deg=d.get('degrade',0)
deg = deg+1 if val > 1.10*d['best'] else 0
d['degrade']=deg
json.dump(d,open(p,'w'),indent=2)
print('VALSTATE',d)
sys.exit(3 if deg>=2 else 0)
PY
  if [ $? -eq 3 ]; then echo "EARLY_STOP_VAL_DEGRADE $(ts)" >> "$LOG"; exit 2; fi

  if [ $CUM -eq 1000 ] || [ $CUM -eq 4000 ]; then
    OUT="/workspace/eval_restore_20261117/reconstructed_normalizer/c_class_eval_v2_step${CUM}"
    mkdir -p "$OUT"
    cp "$EV/A_summary.json" "$EV/episode_A_seed29.json" "$EV/episode_A_seed30.json" \
       "$EV/episode_A_seed29.npz" "$EV/episode_A_seed30.npz" "$OUT/" 2>>"$LOG"
    CK=$(printf '%s/delta_step_%06d.pt' "$PUB" "$CUM")
    C_RESUME=1 C_OUT="$OUT" C_CKPT="$CK" C_SEEDS=29,30 C_MAX_STEPS=600 \
      /opt/venv/bin/python c_class_eval_20261119.py > "eval_step${CUM}.log" 2>&1
    echo "EVAL $CUM rc=$? $(ts)" >> "$LOG"
  fi
done

# final one-shot evaluation on NEVER-used legal seeds 13/16 (11-30 is the screened set)
mkdir -p /workspace/eval_restore_20261117/reconstructed_normalizer/c_class_eval_final_13_16
C_OUT=/workspace/eval_restore_20261117/reconstructed_normalizer/c_class_eval_final_13_16 \
C_CKPT="$PUB/delta_latest.pt" C_SEEDS=13,16 C_MAX_STEPS=600 \
  /opt/venv/bin/python c_class_eval_20261119.py > eval_final_13_16.log 2>&1
echo "FINAL_EVAL rc=$? $(ts)" >> "$LOG"
/opt/venv/bin/python - <<'PY' >> "$LOG" 2>&1
import json, datetime, os, glob
p='MODEL_REPAIR_STATE.json'
d=json.load(open(p))
fin=glob.glob('c_class_eval_final_13_16/summary.json')
d.update({'stage':'segmented_training_complete' if fin else 'segmented_training_no_final_eval',
          'checkpoint':'model_repair_runs/action_expert_v2/delta_latest.pt',
          'training_steps':12000, 'next_task':'analyse_final_c_class',
          'updated':datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')})
json.dump(d,open(p,'w'),indent=2)
print('STATE', d['stage'])
PY
echo "DRIVE_DONE $(ts)" >> "$LOG"
