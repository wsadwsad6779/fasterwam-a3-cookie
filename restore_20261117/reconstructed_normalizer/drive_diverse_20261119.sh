#!/bin/bash
# Bounded adaptation on the DIVERSE teacher corpus (17 layouts, off-grid val).
# 500+500+1000+2000+4000+4000 = up to 12,000 optimizer steps, cosine over 12k.
# Early stop uses the held-out-layout ACTION loss, not the total: the force term is
# high-variance across windows and made the previous run stop for the wrong reason.
# Milestone C-class evals reuse the archived A arm (same protocol, same seeds 29/30).
set -u
cd /workspace/eval_restore_20261117/reconstructed_normalizer || exit 1
LOG=drive_diverse.log
WORK=/root/repair_runs/action_expert_v3
DATA=/workspace/eval_restore_20261117/reconstructed_normalizer/teacher_diverse_20261119
PUB=/workspace/eval_restore_20261117/reconstructed_normalizer/model_repair_runs/action_expert_v3
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
    TRAIN_DATA_DIR=$DATA TOTAL_TARGET=12000 \
    /opt/venv/bin/python train_action_expert_segmented_20261119.py > "seg_div_${CUM}.log" 2>&1
  RC=$?
  echo "SEG to=$CUM rc=$RC $(ts)" >> "$LOG"
  if [ $RC -ne 0 ] || ! grep -q SEGMENT_DONE "seg_div_${CUM}.log"; then
    echo "SEGMENT_FAILED at $CUM $(ts)" >> "$LOG"
    exit 1
  fi
  read VAL ACT <<<"$(tail -n 1 "$WORK/segments.jsonl" | /opt/venv/bin/python -c \
      "import sys,json;d=json.load(sys.stdin);print(d['val_loss'], d['val_action'])")"
  echo "VAL $CUM total=$VAL action=$ACT $(ts)" >> "$LOG"
  /opt/venv/bin/python - "$CUM" "$VAL" "$ACT" <<'PY' >> "$LOG" 2>&1
import json, sys
cum, val, act = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
p='/workspace/eval_restore_20261117/reconstructed_normalizer/drive_div_val.json'
try:
    d = json.load(open(p))
except Exception:
    d = {'history': []}
d['history'].append([cum, val, act])
best = d.get('best_action', 1e9)
d['best_action'] = min(best, act)
best_cum = d.get('best_action_step', cum)
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

  case $CUM in
    1000|2000|4000|8000)
      OUT="/workspace/eval_restore_20261117/reconstructed_normalizer/c_class_eval_v3_step${CUM}"
      mkdir -p "$OUT"
      cp "$EV/A_summary.json" "$EV/episode_A_seed29.json" "$EV/episode_A_seed30.json" \
         "$EV/episode_A_seed29.npz" "$EV/episode_A_seed30.npz" "$OUT/" 2>>"$LOG"
      CK=$(printf '%s/delta_step_%06d.pt' "$WORK" "$CUM")
      [ -f "$CK" ] || { echo "MISSING_CKPT $CK $(ts)" >> "$LOG"; exit 1; }
      C_RESUME=1 C_OUT="$OUT" C_CKPT="$CK" C_SEEDS=29,30 C_MAX_STEPS=600 \
        /opt/venv/bin/python c_class_eval_20261119.py > "eval_div_step${CUM}.log" 2>&1
      echo "EVAL $CUM rc=$? $(ts)" >> "$LOG"
      ;;
  esac
done

# Frozen final seeds are touched exactly once, and only if the development seeds showed
# any cookie entering the target. Otherwise the frozen set stays unused for a later attempt.
BEST=$(/opt/venv/bin/python - <<'PY'
import json
d = json.load(open('/workspace/eval_restore_20261117/reconstructed_normalizer/drive_div_val.json'))
print(d.get('best_action_step', 0))
PY
)
HAS_PROGRESS=$(/opt/venv/bin/python - <<'PY'
import glob, json
ok = False
for p in glob.glob('/workspace/eval_restore_20261117/reconstructed_normalizer/c_class_eval_v3_step*/summary.json'):
    d = json.load(open(p))
    for e in d.get('C', []):
        if e.get('cookies_final', 0) > 0 or e.get('exact_fill_any'):
            ok = True
json.dump({'dev_progress': ok}, open('/workspace/eval_restore_20261117/reconstructed_normalizer/drive_div_dev_progress.json','w'))
print('yes' if ok else 'no')
PY
)
echo "DEV_PROGRESS $HAS_PROGRESS best_step=$BEST $(ts)" >> "$LOG"
if [ "$HAS_PROGRESS" = "yes" ] && [ -f "$WORK/delta_step_$(printf '%06d' "$BEST").pt" ]; then
  OUT=/workspace/eval_restore_20261117/reconstructed_normalizer/c_class_eval_final_13_16
  mkdir -p "$OUT"
  CK="$WORK/delta_step_$(printf '%06d' "$BEST").pt"
  C_OUT="$OUT" C_CKPT="$CK" C_SEEDS=13,16 C_MAX_STEPS=600 \
    /opt/venv/bin/python c_class_eval_20261119.py > eval_div_final_13_16.log 2>&1
  echo "FINAL_EVAL rc=$? $(ts)" >> "$LOG"
  cp "$CK" "$PUB/delta_selected.pt" 2>>"$LOG"
else
  echo "FINAL_EVAL_SKIPPED no_dev_progress $(ts)" >> "$LOG"
fi
/opt/venv/bin/python - <<'PY' >> "$LOG" 2>&1
import json, datetime, os, glob
p='/workspace/eval_restore_20261117/reconstructed_normalizer/MODEL_REPAIR_STATE.json'
d=json.load(open(p))
d.update({'stage':'diverse_corpus_training_finished',
          'diverse_corpus': True,
          'segments_log':'drive_diverse.log',
          'selected_delta': ('model_repair_runs/action_expert_v3/delta_selected.pt'
                             if os.path.exists('/workspace/eval_restore_20261117/reconstructed_normalizer/model_repair_runs/action_expert_v3/delta_selected.pt') else None),
          'final_eval_done': bool(glob.glob('/workspace/eval_restore_20261117/reconstructed_normalizer/c_class_eval_final_13_16/summary.json')),
          'updated':datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')})
json.dump(d,open(p,'w'),indent=2)
print('STATE', d['stage'])
PY
echo "DRIVE_DONE $(ts)" >> "$LOG"