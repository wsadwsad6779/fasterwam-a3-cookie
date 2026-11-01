#!/usr/bin/env bash
# Check 4: does *saving* perturb the trajectory it is saving from?
#
# r4 died 5 seconds after its first light state at step 500:
# `_optimizer_cpu_state` wrote CPU copies back through the dicts returned by
# `Optimizer.state_dict()`, which still reference the live moments, so the next
# `optimizer.step()` raised from `_group_tensors_by_device_and_dtype`.
# Check 3 could not have caught it -- both of its arms stopped right after a
# save, so nothing ever stepped on a poisoned optimizer.
#
#   S: 6 steps with save_every=3  -> saves at 3, then keeps training to 6
#   N: 6 steps with save_every=6  -> saves only at the end, no mid-run save
#
# Same seed, same schedule, same window order, so any difference belongs to the
# save itself.  Nothing outside the two chk4_* output directories is touched.
set -u
source /workspace/activate_fasterwam_a3.sh
export FW_SKIP_STATE=1
export FW_BASE_CKPT=/workspace/a3_train_runs/a3_lora_r3/step_003000_merged.pt
export FW_LORA_INIT_SEED=42

D=/workspace/_r4_stage/chk4
S_DIR=/workspace/a3_train_runs/chk4_S
N_DIR=/workspace/a3_train_runs/chk4_N
export FW_WINDOW_LOG=$D/windows.txt

mkdir -p "$D"
: > "$D/status.txt"
say() { echo "$*" | tee -a "$D/status.txt"; }

if pgrep -f "scripts/train" > /dev/null; then
  say "ABORT a training job is already running"; exit 1
fi
case "$S_DIR$N_DIR" in
  */chk4_S*/chk4_N) rm -rf "$S_DIR" "$N_DIR";;
  *) say "ABORT refusing to touch $S_DIR $N_DIR"; exit 1;;
esac
rm -f "$FW_WINDOW_LOG"

cd /workspace/FasterWAM || exit 1
say "== S: 6 steps, save_every=3 (save then continue) =="
accelerate launch --num_processes 1 scripts/train_light.py \
  data=a3_2cam task=a3_fasterwam batch_size=6 max_steps=6 \
  learning_rate=5e-5 log_every=1 eval_every=0 save_every=3 \
  output_dir="$S_DIR" resume="$FW_BASE_CKPT" > "$D/S.log" 2>&1
say "S_exit=$?  (1 before the light_state fix: RuntimeError in optimizer.step)"

say "== N: 6 steps, save_every=6 (no mid-run save) =="
accelerate launch --num_processes 1 scripts/train_light.py \
  data=a3_2cam task=a3_fasterwam batch_size=6 max_steps=6 \
  learning_rate=5e-5 log_every=1 eval_every=0 save_every=6 \
  output_dir="$N_DIR" resume="$FW_BASE_CKPT" > "$D/N.log" 2>&1
say "N_exit=$?"

/opt/venv/bin/python3 /workspace/_r4_stage/chk3_cmp.py \
  "$S_DIR/checkpoints/light/step_000006" \
  "$N_DIR/checkpoints/light/step_000006" \
  "$D/S.log" "$D/N.log" > "$D/report_S_vs_N.txt" 2>&1
say "CMP_exit=$? report=$D/report_S_vs_N.txt"
say "window_log_lines=$(wc -l < "$FW_WINDOW_LOG" 2>/dev/null || echo missing)"
say "ALL_DONE"
