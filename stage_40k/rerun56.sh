#!/usr/bin/env bash
# Re-run rehearsal stages 5 and 6 only.
#
# Stages 0-4 already produced their artifacts (the 6/12-step run, the light
# states, the ledgers and the 11 GB merged file), and rehearsal65.sh deletes all
# of them at startup. The only things that changed are the two checkers, so this
# replays the two verification stages against the existing evidence instead of
# burning another 11 GB write.
set -u

STAGE=/workspace/_v1_stage
FW=$STAGE/fw
LED=$STAGE/rehearsal65
PY=/opt/venv/bin/python
BASE=/workspace/FasterWAM/checkpoints/fasterwam_release/libero/step_021700.pt
RUN=$LED/run
MERGED=$LED/step_000012_merged.pt
ARGS=(data=a3_2cam_force task=a3_force_40k batch_size=6 max_steps=40000
      learning_rate=1e-4 log_every=1 eval_every=0)

cd "$FW" || { echo "@@ABORT@@ no fw dir"; exit 9; }
export PYTHONPATH=$FW/src
export FW_BASE_CKPT=$BASE
export FW_SEG_LEDGER=$LED/seg_ledger.jsonl
export FW_BASE_LOAD_LEDGER=$LED/base_load_ledger.jsonl
export FW_LORA_INIT_SEED=42
export FW_VERIFY_BASE_LOAD=1
export FASTERWAM_A3_INSTRUCTION="pick up the cookie and place it in the box"
export TOKENIZERS_PARALLELISM=false
test -s "$MERGED" || { echo "@@ABORT@@ no merged file"; exit 9; }

echo "@@STAGE_5@@ reload the published file the way deployment does"
export FW_LEDGER_STAGE=publish_verify
$PY -u scripts/verify_publish65.py "${ARGS[@]}" output_dir="$LED" \
  +base_ckpt="$BASE" +light_dir="$RUN/checkpoints/light/step_000012" \
  +merged="$MERGED" +calib=/workspace/a3_data/a3_fw_force/force_calib_v1.json \
  +report="$LED"/publish_verify.json > "$LED"/publish_verify.log 2>&1
echo "@@STAGE_5_RC@@ $?"

echo "@@STAGE_6@@ check the ledgers, LR ladder, sampler continuity and the budget"
$PY "$STAGE"/verify_rehearsal65.py --led "$LED" --report "$LED"/rehearsal_report.json \
  > "$LED"/verify.log 2>&1
echo "@@STAGE_6_RC@@ $?"
echo "@@DONE@@"
