#!/usr/bin/env bash
# r4: 4,000-step LoRA warm restart on top of the r3 merged checkpoint.
#
# Approved scope: the original 100 demonstrations only, peak lr 5e-5 cosine,
# `FW_SAMPLE_OFFSET=18000` so the first batch is window 18,000 of the r2/r3
# permutation instead of a third replay of 0..17,999, and light-state saves
# every 500 steps (adapter + optimizer + scheduler + RNG + counters, ~185 MB).
# r3 -> r4 is a *warm restart*: r3 kept no optimizer state, so no claim of a
# bit-exact continuation of r3 is possible or made here.
set -u
source /workspace/activate_fasterwam_a3.sh

BASE=/workspace/a3_train_runs/a3_lora_r3/step_003000_merged.pt
OUT=/workspace/a3_train_runs/a3_lora_r4
export FW_SKIP_STATE=1
export FW_BASE_CKPT=$BASE
export FW_LORA_INIT_SEED=42
export FW_SAMPLE_OFFSET=18000
export FW_WINDOW_LOG=$OUT/windows.log

say() { echo "[$(date -u +%FT%TZ)] $*"; }

# ---- preflight: refuse to start anything that could damage an earlier run ----
[ -f "$BASE" ] || { say "ABORT base missing: $BASE"; exit 1; }
if pgrep -f "scripts/train" > /dev/null; then
  say "ABORT a trainer is already running"; exit 1
fi
if [ -e "$OUT" ]; then
  say "ABORT $OUT already exists -- r4 must not overwrite an earlier run"; exit 1
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
  echo "base=$BASE"
  echo "base_size_bytes=$(stat -c %s "$BASE")"
  echo "base_sha256_head=$(head -c 4194304 "$BASE" | sha256sum | cut -d' ' -f1)"
  echo "r4_patch_sha256=$(sha256sum /workspace/FasterWAM/src/fasterwam/r4_patch.py | cut -d' ' -f1)"
  echo "light_state_sha256=$(sha256sum /workspace/FasterWAM/src/fasterwam/light_state.py | cut -d' ' -f1)"
  echo "avail_bytes_before=$AVAIL need=$NEED"
  echo "FW_SAMPLE_OFFSET=$FW_SAMPLE_OFFSET FW_LORA_INIT_SEED=$FW_LORA_INIT_SEED FW_SKIP_STATE=1"
  echo "hydra: data=a3_2cam task=a3_fasterwam batch_size=6 max_steps=4000"
  echo "       learning_rate=5e-5 log_every=20 save_every=500 eval_every=0"
} > "$OUT/preflight.txt" 2>&1
cat "$OUT/preflight.txt"

cd /workspace/FasterWAM || exit 1
say "r4 start"
accelerate launch --num_processes 1 scripts/train_light.py \
  data=a3_2cam task=a3_fasterwam batch_size=6 max_steps=4000 \
  learning_rate=5e-5 log_every=20 save_every=500 eval_every=0 \
  output_dir="$OUT" resume="$BASE" > "$OUT/train.log" 2>&1
RC=$?
say "r4 exit=$RC"
echo "exit_code=$RC" >> "$OUT/preflight.txt"
echo "avail_bytes_after=$(df -B1 --output=avail /workspace | tail -1 | tr -d ' ')" >> "$OUT/preflight.txt"
exit $RC
