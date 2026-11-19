#!/bin/bash
# v4 continuation: 12000 -> 16000 steps on the diverse teacher corpus, with
# gripper-aware window oversampling (GRIP_REPLICATE=8) against the always-close
# collapse of action dim 6. Resumes from the v3 step-12000 last.pt.
# C-class milestone evals (seeds 29/30, WITH video) at 13000/14000/15000/16000.
# Frozen seeds 13/16 are touched at most once, only if a dev eval shows progress.
set -u
cd /workspace/eval_restore_20261117/reconstructed_normalizer || exit 1
LOG=drive_v4.log
WORK=/root/repair_runs/action_expert_v4
V3=/root/repair_runs/action_expert_v3
DATA=/workspace/eval_restore_20261117/reconstructed_normalizer/teacher_diverse_20261119
PUB=/workspace/eval_restore_20261117/reconstructed_normalizer/model_repair_runs/action_expert_v4
EV=/workspace/eval_restore_20261117/reconstructed_normalizer/c_class_eval_v3_step8000
ts() { date -u +%FT%TZ; }
echo "V4_START $(ts)" >> "$LOG"

mkdir -p "$WORK" "$PUB"
if [ ! -f "$WORK/last.pt" ]; then
  cp "$V3/last.pt" "$WORK/last.pt" || { echo "NO_V3_LAST $(ts)" >> "$LOG"; exit 1; }
fi

CUM=12000
for S in 1000 1000 1000 1000; do
  CUM=$((CUM + S))
  SEG_STEPS=$S TRAIN_RESUME=1 TRAIN_WORK=$WORK TRAIN_PUB=$PUB \
    TRAIN_DATA_DIR=$DATA TOTAL_TARGET=16000 TRAIN_LR=5e-5 GRIP_REPLICATE=8 \
    /opt/venv/bin/python train_action_expert_v4_20261119.py > "seg_v4_${CUM}.log" 2>&1
  RC=$?
  echo "SEG to=$CUM rc=$RC $(ts)" >> "$LOG"
  if [ $RC -ne 0 ] || ! grep -q SEGMENT_DONE "seg_v4_${CUM}.log"; then
    echo "SEGMENT_FAILED at $CUM $(ts)" >> "$LOG"
    exit 1
  fi
  read VAL ACT <<<"$(tail -n 1 "$WORK/segments.jsonl" | /opt/venv/bin/python -c \
      "import sys,json;d=json.load(sys.stdin);print(d['val_loss'], d['val_action'])")"
  echo "VAL $CUM total=$VAL action=$ACT $(ts)" >> "$LOG"
  /opt/venv/bin/python - "$CUM" "$VAL" "$ACT" <<'PY' >> "$LOG" 2>&1
import json, sys
cum, val, act = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
p='/workspace/eval_restore_20261117/reconstructed_normalizer/drive_v4_val.json'
try:
    d = json.load(open(p))
except Exception:
    d = {'history': []}
d['history'].append([cum, val, act])
best = d.get('best_action', 1e9)
d['best_action'] = min(best, act)
if act <= best:
    d['best_action_step'] = cum
deg = d.get('degrade', 0)
deg = deg + 1 if act > 1.25 * d['best_action'] else 0
d['degrade'] = deg
json.dump(d, open(p, 'w'), indent=2)
print('VALSTATE', d)
sys.exit(3 if deg >= 2 else 0)
PY
  if [ $? -eq 3 ]; then echo "EARLY_STOP_VAL_ACTION_DEGRADE $(ts)" >> "$LOG"; break; fi

  OUT="/workspace/eval_restore_20261117/reconstructed_normalizer/c_class_eval_v4_step${CUM}"
  mkdir -p "$OUT"
  cp "$EV/A_summary.json" "$EV/episode_A_seed29.json" "$EV/episode_A_seed30.json" \
     "$EV/episode_A_seed29.npz" "$EV/episode_A_seed30.npz" "$OUT/" 2>>"$LOG"
  CK=$(printf '%s/delta_step_%06d.pt' "$WORK" "$CUM")
  [ -f "$CK" ] || { echo "MISSING_CKPT $CK $(ts)" >> "$LOG"; exit 1; }
  C_RESUME=1 C_OUT="$OUT" C_CKPT="$CK" C_SEEDS=29,30 C_MAX_STEPS=600 C_VIDEO=1 \
    /opt/venv/bin/python c_class_eval_20261119.py > "eval_v4_step${CUM}.log" 2>&1
  echo "EVAL $CUM rc=$? $(ts)" >> "$LOG"
  grep -E '\[C seed' "eval_v4_step${CUM}.log" >> "$LOG" 2>&1
done

# Frozen final seeds: touched exactly once, only on demonstrated dev progress.
BEST=$(/opt/venv/bin/python - <<'PY'
import json
d = json.load(open('/workspace/eval_restore_20261117/reconstructed_normalizer/drive_v4_val.json'))
print(d.get('best_action_step', 0))
PY
)
HAS_PROGRESS=$(/opt/venv/bin/python - <<'PY'
import glob, json
ok = False
for p in glob.glob('/workspace/eval_restore_20261117/reconstructed_normalizer/c_class_eval_v4_step*/summary.json'):
    d = json.load(open(p))
    for e in d.get('C', []):
        if e.get('cookies_final', 0) > 0 or e.get('exact_fill_any'):
            ok = True
json.dump({'dev_progress': ok}, open('/workspace/eval_restore_20261117/reconstructed_normalizer/drive_v4_dev_progress.json','w'))
print('yes' if ok else 'no')
PY
)
echo "DEV_PROGRESS $HAS_PROGRESS best_step=$BEST $(ts)" >> "$LOG"
if [ "$HAS_PROGRESS" = "yes" ]; then
  # touch 13/16 at the best dev milestone
  BK=$BEST
  OUT=/workspace/eval_restore_20261117/reconstructed_normalizer/c_class_eval_v4_final_13_16
  mkdir -p "$OUT"
  CK="$WORK/delta_step_$(printf '%06d' "$BK").pt"
  if [ -f "$CK" ]; then
    C_OUT="$OUT" C_CKPT="$CK" C_SEEDS=13,16 C_MAX_STEPS=600 C_VIDEO=1 \
      /opt/venv/bin/python c_class_eval_20261119.py > eval_v4_final_13_16.log 2>&1
    echo "FINAL_EVAL rc=$? $(ts)" >> "$LOG"
    cp "$CK" "$PUB/delta_selected.pt" 2>>"$LOG"
  else
    echo "FINAL_EVAL_SKIPPED missing_ckpt $(ts)" >> "$LOG"
  fi
else
  echo "FINAL_EVAL_SKIPPED no_dev_progress $(ts)" >> "$LOG"
fi
/opt/venv/bin/python - <<'PY' >> "$LOG" 2>&1
import json, datetime, os, glob
p='/workspace/eval_restore_20261117/reconstructed_normalizer/MODEL_REPAIR_STATE.json'
d=json.load(open(p))
d.update({'stage':'v4_gripper_oversample_training_finished',
          'v4_drive_log':'drive_v4.log',
          'v4_work':'/root/repair_runs/action_expert_v4',
          'v4_val':'drive_v4_val.json',
          'v4_dev_progress': json.load(open('/workspace/eval_restore_20261117/reconstructed_normalizer/drive_v4_dev_progress.json')) if os.path.exists('/workspace/eval_restore_20261117/reconstructed_normalizer/drive_v4_dev_progress.json') else None,
          'final_eval_done': bool(glob.glob('/workspace/eval_restore_20261117/reconstructed_normalizer/c_class_eval_v4_final_13_16/summary.json')),
          'updated':datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')})
json.dump(d,open(p,'w'),indent=2)
print('STATE', d['stage'])
PY
echo "V4_DONE $(ts)" >> "$LOG"
