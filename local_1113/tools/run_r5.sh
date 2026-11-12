#!/usr/bin/env bash
# r5: continue the r4 LoRA run for 4,000 more optimiser steps (trainer counter
# 4,000 -> 8,000; segment sum 2,000 + 3,000 + 3,000 + 4,000 + 4,000 = 16,000).
#
# This is a true resume, not a warm restart: r4 wrote a light state holding the
# adapter, the AdamW moments, the scheduler, the RNG streams and the sampler
# position, and `r4_patch` restores all of them. The single deliberate change is
# the learning-rate schedule -- r4's cosine is already annealed to its floor, so
# `FW_LR_RESTART=1` rebuilds the same warmup+cosine shape across the new 4,000
# steps and records both the restored and the restarted schedule in
# lr_restart.json. See fasterwam/r5_patch.py.
#
# Unchanged on purpose: the same 100 demonstrations / 39,481 windows, batch 6,
# bf16, LoRA r=16 alpha=32 on q,k,v,o, no new wrapper or safety rule, no change
# to the action representation or the input structure, light-state saves every
# 500 steps, and r4's directory is read-only from here.
set -u
source /workspace/activate_fasterwam_a3.sh

BASE=/workspace/a3_train_runs/a3_lora_r3/step_003000_merged.pt
R4LIGHT=/workspace/a3_train_runs/a3_lora_r4/checkpoints/light/step_004000
OUT=/workspace/a3_train_runs/a3_lora_r5
export FW_SKIP_STATE=1
export FW_BASE_CKPT=$BASE
export FW_LORA_INIT_SEED=42
export FW_LR_RESTART=1
export FW_WINDOW_LOG=$OUT/windows.log

say() { echo "[$(date -u +%FT%TZ)] $*"; }

# ---- preflight: refuse to start anything that could damage an earlier run ----
[ -f "$BASE" ]          || { say "ABORT base missing: $BASE"; exit 1; }
[ -f "$R4LIGHT/trainer_state.json" ] || { say "ABORT r4 light state missing"; exit 1; }
for f in adapter.pt optimizer.pt scheduler.pt rng.pt; do
  [ -f "$R4LIGHT/$f" ] || { say "ABORT r4 state file missing: $f"; exit 1; }
done
if pgrep -f "scripts/train" > /dev/null; then
  say "ABORT a trainer is already running"; exit 1
fi
if [ -e "$OUT" ]; then
  say "ABORT $OUT already exists -- r5 must not overwrite an earlier run"; exit 1
fi
AVAIL=$(df -B1 --output=avail /workspace | tail -1 | tr -d ' ')
NEED=2500000000            # 8 light states (185 MB each) + window log + margin
if [ "$AVAIL" -le "$NEED" ]; then
  say "ABORT only $AVAIL bytes free on /workspace, need $NEED"; exit 1
fi

mkdir -p "$OUT"
{
  say "preflight ok"
  echo "output_dir=$OUT"
  echo "resume_light_state=$R4LIGHT"
  echo "FW_BASE_CKPT=$BASE"
  echo "base_size_bytes=$(stat -c %s "$BASE")"
  echo "base_sha256_head=$(head -c 4194304 "$BASE" | sha256sum | cut -d' ' -f1)"
  echo "--- r4 light state being resumed from (read-only) ---"
  for f in adapter.pt optimizer.pt scheduler.pt rng.pt trainer_state.json; do
    echo "$f $(stat -c %s "$R4LIGHT/$f") $(sha256sum "$R4LIGHT/$f" | cut -d' ' -f1)"
  done
  sed -e 's/^/  /' "$R4LIGHT/trainer_state.json"
  echo "--- patch/source identity ---"
  for f in r4_patch.py r5_patch.py light_state.py trainer.py; do
    echo "$f $(sha256sum /workspace/FasterWAM/src/fasterwam/$f | cut -d' ' -f1)"
  done
  echo "train_light_r5.py $(sha256sum /workspace/FasterWAM/scripts/train_light_r5.py | cut -d' ' -f1)"
  echo "avail_bytes_before=$AVAIL need=$NEED"
  echo "FW_LR_RESTART=1 FW_LORA_INIT_SEED=42 FW_SKIP_STATE=1 (FW_SAMPLE_OFFSET unset)"
  echo "hydra: data=a3_2cam task=a3_fasterwam batch_size=6 max_steps=8000"
  echo "       learning_rate=5e-5 log_every=20 save_every=500 eval_every=0"
  echo "segments: r1=2000 r2=3000 r3=3000 r4=4000 -> 12000; r5=+4000 -> 16000"
} > "$OUT/preflight.txt" 2>&1
cat "$OUT/preflight.txt"

cd /workspace/FasterWAM || exit 1
say "r5 start"
accelerate launch --num_processes 1 scripts/train_light_r5.py \
  data=a3_2cam task=a3_fasterwam batch_size=6 max_steps=8000 \
  learning_rate=5e-5 log_every=20 save_every=500 eval_every=0 \
  output_dir="$OUT" resume="$R4LIGHT" > "$OUT/train.log" 2>&1
RC=$?
say "r5 exit=$RC"
echo "exit_code=$RC" >> "$OUT/preflight.txt"
echo "avail_bytes_after=$(df -B1 --output=avail /workspace | tail -1 | tr -d ' ')" >> "$OUT/preflight.txt"
exit $RC
