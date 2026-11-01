#!/usr/bin/env bash
# r5 dry run: resume the r4 light state and train exactly two optimiser steps
# into a scratch directory, to prove the restore is real before the 2h40 run.
#
# What this has to show, and where it shows it:
#   global_step / epoch / batch_in_epoch  -> "Resumed light state ... step=4000"
#   sampler position                      -> "sampler pass=1 windows=36967 head=H"
#       and H must be the window r4's own windows.log lists at pass-1 batch
#       419 * 6, i.e. the run continues the permutation instead of restarting it
#   optimizer moments                     -> step=4002 in the saved optimizer
#   scheduler                             -> lr_restart.json + lr= in the log
#   rng                                   -> rng.pt restored; roundtrip check is
#       a separate CPU-only script
#   light checkpoint                      -> /workspace/a3_train_runs/_r5_dryrun/
#       checkpoints/light/step_004002
set -u
source /workspace/activate_fasterwam_a3.sh

BASE=/workspace/a3_train_runs/a3_lora_r3/step_003000_merged.pt
R4=/workspace/a3_train_runs/a3_lora_r4
R4LIGHT=$R4/checkpoints/light/step_004000
OUT=/workspace/a3_train_runs/_r5_dryrun
export FW_SKIP_STATE=1
export FW_BASE_CKPT=$BASE
export FW_LORA_INIT_SEED=42
export FW_LR_RESTART=1
export FW_WINDOW_LOG=$OUT/windows.log

if pgrep -f "scripts/train" > /dev/null; then
  echo "[$(date -u +%FT%TZ)] ABORT a trainer is already running"; exit 1
fi
if [ -e "$OUT" ]; then
  rm -rf "$OUT"          # scratch only; nothing upstream is written here
fi
mkdir -p "$OUT"

cd /workspace/FasterWAM || exit 1
echo "[$(date -u +%FT%TZ)] r5 dry run start"
accelerate launch --num_processes 1 scripts/train_light_r5.py \
  data=a3_2cam task=a3_fasterwam batch_size=6 max_steps=4002 \
  learning_rate=5e-5 log_every=1 save_every=1 eval_every=0 \
  output_dir="$OUT" resume="$R4LIGHT" > "$OUT/train.log" 2>&1
RC=$?
echo "[$(date -u +%FT%TZ)] r5 dry run exit=$RC"

{
  echo "==== resume / sampler / restart lines"
  grep -n -e "Resumed light state" -e "sampler pass" -e "LR restart" \
        -e "Loading frozen base" -e "Traceback" -e "Error" "$OUT/train.log"
  echo "==== first optimiser steps (lr must start near 2.5e-07, not 5e-05)"
  grep -n "step=400" "$OUT/train.log" | head -20
  echo "==== lr_restart.json"
  cat "$OUT/lr_restart.json"
  echo "==== saved trainer_state.json at step_004002"
  cat "$OUT/checkpoints/light/step_004002/trainer_state.json"
  echo "==== optimizer step counter carried from r4"
  python3 - <<'PY'
import torch
for tag, path in [("r4 ", "/workspace/a3_train_runs/a3_lora_r4/checkpoints/light/step_004000/optimizer.pt"),
                  ("r5d", "/workspace/a3_train_runs/_r5_dryrun/checkpoints/light/step_004002/optimizer.pt")]:
    st = torch.load(path, map_location="cpu", weights_only=False)
    steps = sorted({v["step"] for v in st["state"].values() if "step" in v})
    groups = [{k: g[k] for k in ("lr", "initial_lr", "betas", "weight_decay") if k in g}
              for g in st["param_groups"]]
    print(tag, "params_with_state=%d step_values=%s" % (len(st["state"]), steps))
    print("   groups=%s" % groups[:2])
PY
  echo "==== window continuation: r4 pass-1 line 2515 vs r5 dry-run pass-1 head"
  python3 - <<'PY'
r4 = [l.split() for l in open("/workspace/a3_train_runs/a3_lora_r4/windows.log")
      if not l.startswith("#")]
p1 = [int(w) for p, w in r4 if int(p) == 1]
print("r4 pass-1 logged windows: %d  first=%s  2515th=%s" % (len(p1), p1[0], p1[2514]))
d5 = [l.split() for l in open("/workspace/a3_train_runs/_r5_dryrun/windows.log")
      if not l.startswith("#")]
p5 = [int(w) for p, w in d5 if int(p) == 1]
print("r5 dry-run first logged windows: %s" % p5[:6])
print("CONTIGUOUS=%s" % (p5[:6] == p1[2514:2520]))
PY
  ls -la "$OUT/checkpoints/light/step_004002"
  echo "avail_bytes_after=$(df -B1 --output=avail /workspace | tail -1 | tr -d ' ')"
} > "$OUT/verify.log" 2>&1
cat "$OUT/verify.log"
exit $RC
