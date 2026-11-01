#!/usr/bin/env bash
# Check 3, control arm: a second uninterrupted 10-step run.
#
# Run 2 of chk3_run.sh already showed that with FW_LORA_INIT_SEED the resumed
# run reproduces the continuous one to within one bf16 ulp.  This answers the
# only remaining question -- is that residual the resume path or the GPU?  Two
# *identical* continuous runs (A from chk3_run.sh, A2 here) differ by the same
# amount and on the same handful of elements, then nothing in the save/restore
# seam is at fault.  Also the first run made with the shipped r4_patch.py, which
# writes the sampler order to FW_WINDOW_LOG.
set -u
source /workspace/activate_fasterwam_a3.sh
export FW_SKIP_STATE=1
export FW_BASE_CKPT=/workspace/a3_train_runs/a3_lora_r3/step_003000_merged.pt
export FW_LORA_INIT_SEED=42

D=/workspace/_r4_stage/chk3
A_DIR=/workspace/a3_train_runs/chk3_A
A2_DIR=/workspace/a3_train_runs/chk3_A2
export FW_WINDOW_LOG=$D/windows_a2.txt

mkdir -p "$D"
: > "$D/status_a2.txt"
say() { echo "$*" | tee -a "$D/status_a2.txt"; }

if pgrep -f "scripts/train" > /dev/null; then
  say "ABORT a training job is already running"; exit 1
fi
case "$A2_DIR" in
  */chk3_A2) rm -rf "$A2_DIR";;
  *) say "ABORT refusing to touch $A2_DIR"; exit 1;;
esac
rm -f "$FW_WINDOW_LOG"

cd /workspace/FasterWAM || exit 1
say "== A2: 10 continuous steps, shipped r4_patch =="
accelerate launch --num_processes 1 scripts/train_light.py \
  data=a3_2cam task=a3_fasterwam batch_size=6 max_steps=10 \
  learning_rate=5e-5 log_every=1 eval_every=0 save_every=10 \
  output_dir="$A2_DIR" resume="$FW_BASE_CKPT" > "$D/A2.log" 2>&1
say "A2_exit=$?"

/opt/venv/bin/python3 /workspace/_r4_stage/chk3_cmp.py \
  "$A_DIR/checkpoints/light/step_000010" \
  "$A2_DIR/checkpoints/light/step_000010" \
  "$D/A.log" "$D/A2.log" > "$D/report_A_vs_A2.txt" 2>&1
say "CMP_exit=$? report=$D/report_A_vs_A2.txt"
say "window_log_lines=$(wc -l < "$FW_WINDOW_LOG" 2>/dev/null || echo missing)"
say "ALL_DONE"
