#!/bin/bash
# Bounded v2 chain: corrected-data action-expert fine-tune (300 steps) -> C-class eval on
# frozen test seeds 29/30. Stops on any failure; every stage appends to the run log.
set -u
cd /workspace/eval_restore_20261117/reconstructed_normalizer || exit 1
LOG=model_repair_v2_chain.log
STATE=MODEL_REPAIR_STATE.json
ts() { date -u +%FT%TZ; }
echo "CHAIN_START $(ts)" >> "$LOG"

/opt/venv/bin/python -m py_compile train_action_expert_300_v2.py || {
  echo "COMPILE_FAIL $(ts)" >> "$LOG"; exit 1; }

TRAIN_OUT=/workspace/eval_restore_20261117/reconstructed_normalizer/model_repair_runs/action_expert_300_v2 \
  /opt/venv/bin/python train_action_expert_300_v2.py > train_v2.log 2>&1
RC=$?
echo "TRAIN_RC=$RC $(ts)" >> "$LOG"
if [ $RC -ne 0 ] || ! grep -q TRAIN_DONE train_v2.log; then
  echo "TRAIN_FAILED $(ts)" >> "$LOG"
  /opt/venv/bin/python - <<'PY'
import json
p='MODEL_REPAIR_STATE.json'
d=json.load(open(p))
d.update({'stage':'action_expert_v2_train_failed','last_error':'train_v2.log',
          'updated':__import__('datetime').datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')})
json.dump(d,open(p,'w'),indent=2)
PY
  exit 1
fi

mkdir -p c_class_eval_v2
cp c_class_eval_20261119/A_summary.json \
   c_class_eval_20261119/episode_A_seed29.json c_class_eval_20261119/episode_A_seed30.json \
   c_class_eval_20261119/episode_A_seed29.npz c_class_eval_20261119/episode_A_seed30.npz \
   c_class_eval_v2/ 2>>"$LOG"

C_RESUME=1 \
C_OUT=/workspace/eval_restore_20261117/reconstructed_normalizer/c_class_eval_v2 \
C_CKPT=/workspace/eval_restore_20261117/reconstructed_normalizer/model_repair_runs/action_expert_300_v2/delta_final.pt \
C_SEEDS=29,30 C_MAX_STEPS=600 \
  /opt/venv/bin/python c_class_eval_20261119.py > c_class_eval_v2.log 2>&1
RC=$?
echo "EVAL_RC=$RC $(ts)" >> "$LOG"
/opt/venv/bin/python - <<'PY' >> "$LOG" 2>&1
import json, datetime
p='MODEL_REPAIR_STATE.json'
d=json.load(open(p))
d.update({'stage':'c_class_v2_done' if __import__('os').path.exists(
              'c_class_eval_v2/summary.json') else 'c_class_v2_failed',
          'checkpoint':'model_repair_runs/action_expert_300_v2/delta_final.pt',
          'training_steps':300,
          'next_task':'analyse_c_class_v2',
          'updated':datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')})
json.dump(d,open(p,'w'),indent=2)
print('STATE_UPDATED', d['stage'])
PY
echo "CHAIN_DONE $(ts)" >> "$LOG"
