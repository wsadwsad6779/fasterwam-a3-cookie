#!/bin/bash
# Run inside JupyterLab Terminal after signing in again. Read-only status first,
# then prints the exact restart commands if a component is missing.
echo "== tmux =="
tmux ls 2>&1
echo "== processes =="
pgrep -af "_t7_train|_t7_chain|_t7_watchdog|_t7_eval|_t7_finalizer" || true
echo "== last train line =="
tail -3 /workspace/fwv1/logs/t7_train.log 2>/dev/null
echo "== checkpoints =="
ls -la /workspace/fwv1/train_runs/t7_21k 2>/dev/null
echo "== milestone summaries =="
ls /workspace/fwv1/eval_fingertip_v1/step_*/SUMMARY_step_*.json 2>/dev/null
echo "== disk =="
df -h /workspace | tail -1
echo "== recommended actions =="
if ! tmux has-session -t fwv1 2>/dev/null; then
  echo "tmux session fwv1 missing:"
  echo "  tmux new -s fwv1 -d 'bash /workspace/_t7_chain.sh'"
fi
if ! pgrep -f "_t7_train.py" > /dev/null && [ ! -f /workspace/fwv1/train_runs/t7_21k/step_003000.pt ]; then
  echo "training not running and 3k checkpoint missing:"
  echo "  T7_RESUME=/workspace/fwv1/train_runs/t7_21k/latest.pt T7_END_STEP=3000 T7_CKPT_EVERY=200 /opt/venv/bin/python /workspace/_t7_train.py"
  echo "  (if latest.pt is missing, use T7_START_STEP=<last metrics step> instead of T7_RESUME)"
fi
if ! pgrep -f "_t7_chain.sh" > /dev/null; then
  echo "milestone chain missing:"
  echo "  tmux new-window -t fwv1 -n chain 'bash /workspace/_t7_chain.sh'"
fi
if ! pgrep -f "_t7_watchdog.sh" > /dev/null; then
  echo "watchdog missing:"
  echo "  tmux new-window -t fwv1 -n watchdog 'bash /workspace/_t7_watchdog.sh'"
fi
if ! pgrep -f "_t7_finalizer.sh" > /dev/null; then
  echo "finalizer missing (only needed before 21k):"
  echo "  tmux new-window -t fwv1 -n finalizer 'bash /workspace/_t7_finalizer.sh'"
fi