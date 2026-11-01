#!/usr/bin/env bash
# Check 3: real-trainer interrupt/resume equivalence.
#
#   A: 10 uninterrupted steps
#   B: the same 10-step schedule, killed right after the step-5 light state is
#      written, then resumed from that directory to step 10
#
# Both use max_steps=10, so the cosine horizon is the same and a resumed run has
# no excuse to differ in LR.  Nothing outside the two chk3_* output directories
# is written or removed.
set -u
source /workspace/activate_fasterwam_a3.sh
export FW_SKIP_STATE=1
export FW_BASE_CKPT=/workspace/a3_train_runs/a3_lora_r3/step_003000_merged.pt

D=/workspace/_r4_stage/chk3
RUNS=/workspace/a3_train_runs
A_DIR=$RUNS/chk3_A
B_DIR=$RUNS/chk3_B
B2_DIR=$RUNS/chk3_B2
COMMON="data=a3_2cam task=a3_fasterwam batch_size=6 max_steps=10 learning_rate=5e-5 log_every=1 eval_every=0"

mkdir -p "$D"
: > "$D/status.txt"
say() { echo "$*" | tee -a "$D/status.txt"; }

if pgrep -f "run_matrix.py|scripts/train" > /dev/null; then
  say "ABORT another job is running"; exit 1
fi
for d in "$A_DIR" "$B_DIR" "$B2_DIR"; do
  case "$d" in
    "$RUNS"/chk3_*) rm -rf "$d";;
    *) say "ABORT refusing to touch $d"; exit 1;;
  esac
done
rm -rf "$D"/* ; mkdir -p "$D"

cd /workspace/FasterWAM || exit 1

say "== A: 10 continuous steps =="
accelerate launch --num_processes 1 scripts/train_light.py $COMMON \
  save_every=10 output_dir="$A_DIR" resume="$FW_BASE_CKPT" > "$D/A.log" 2>&1
say "A_exit=$?"

say "== B: same schedule, save at 5, hard kill after the step-5 state =="
setsid accelerate launch --num_processes 1 scripts/train_light.py $COMMON \
  save_every=5 output_dir="$B_DIR" resume="$FW_BASE_CKPT" > "$D/B1.log" 2>&1 &
BPID=$!
STATE5="$B_DIR/checkpoints/light/step_000005/trainer_state.json"
waited=0
while [ ! -f "$STATE5" ]; do
  sleep 2; waited=$((waited + 2))
  if ! kill -0 "$BPID" 2>/dev/null; then say "ABORT B exited before saving step 5"; exit 1; fi
  if [ "$waited" -gt 1800 ]; then say "ABORT B never reached step 5"; kill -TERM -"$BPID"; exit 1; fi
done
sleep 4                       # let it run into step 6 before the kill
kill -TERM -"$BPID" 2>/dev/null
sleep 10
kill -KILL -"$BPID" 2>/dev/null
wait "$BPID" 2>/dev/null
say "B_killed_after_step5_save waited=${waited}s last_b1_step=$(grep -c 'step=' "$D/B1.log")"

say "== B2: resume from the step-5 light state to step 10 =="
accelerate launch --num_processes 1 scripts/train_light.py $COMMON \
  save_every=5 output_dir="$B2_DIR" \
  resume="$B_DIR/checkpoints/light/step_000005" > "$D/B2.log" 2>&1
say "B2_exit=$?"

/opt/venv/bin/python3 /workspace/_r4_stage/chk3_cmp.py \
  "$A_DIR/checkpoints/light/step_000010" \
  "$B2_DIR/checkpoints/light/step_000010" \
  "$D/A.log" "$D/B1.log" "$D/B2.log" > "$D/report.txt" 2>&1
say "CMP_exit=$? report=$D/report.txt"
say "ALL_DONE"
