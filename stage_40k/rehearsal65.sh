#!/usr/bin/env bash
# Task #65 -- 6-step real-trainer rehearsal for the segmented 40k run.
#
# This is NOT the 40k run. FW_STOP_STEP caps each segment at 6 optimizer steps,
# so the rehearsal costs a few minutes of GPU and ~11 GB of disk while
# exercising the four things a stand-in cannot: the real trainer loop under
# FW_STOP_STEP, the real base_load_guard against the real 11 GB base, the real
# v3 light-state resume, and the real publication path to a merged checkpoint.
#
# Plain `python` (not `accelerate launch`) matches how r4/r5 ran: the trainer
# builds its own Accelerator, so with no launcher env it is NO distributed,
# world_size=1, bf16 -- see trainer.py:59-63. The one accelerate config in the
# repo selects DeepSpeed, which r4/r5 did not use.
#
# Stages write one log each under rehearsal65/ plus a "@@STAGE_n_RC@@ n" marker
# in rehearsal65/driver.log. Every stage runs even if the previous failed, so one
# pass of this script produces the complete failure picture.
set -u

STAGE=/workspace/_v1_stage
FW=$STAGE/fw
LED=$STAGE/rehearsal65
PY=/opt/venv/bin/python
BASE=/workspace/FasterWAM/checkpoints/fasterwam_release/libero/step_021700.pt
RUN=$LED/run
MERGED=$LED/step_000012_merged.pt
PUBREC=$LED/publish_record.json
ARGS=(data=a3_2cam_force task=a3_force_40k batch_size=6 max_steps=40000
      learning_rate=1e-4 log_every=1 eval_every=0)

mkdir -p "$LED"
rm -f "$LED"/seg_ledger.jsonl "$LED"/base_load_ledger.jsonl \
      "$LED"/windows_seg1.log "$LED"/windows_seg2.log "$LED"/sizes.txt \
      "$LED"/rehearsal_report.json "$LED"/publish_verify.json "$PUBREC" \
      "$LED"/dataset_force_verify.json "$MERGED" "$MERGED.tmp"
rm -rf "$RUN" "$LED"/_guard_controls

cd "$FW" || { echo "@@ABORT@@ no fw dir"; exit 9; }
export PYTHONPATH=$FW/src
export FW_BASE_CKPT=$BASE
export FW_SEG_LEDGER=$LED/seg_ledger.jsonl
export FW_BASE_LOAD_LEDGER=$LED/base_load_ledger.jsonl
export FW_LORA_INIT_SEED=42
export FW_VERIFY_BASE_LOAD=1
export FASTERWAM_A3_INSTRUCTION="pick up the cookie and place it in the box"
export TOKENIZERS_PARALLELISM=false

echo "@@STAGE_0@@ preflight: config composes, imports resolve, base exists"
ls -la "$BASE"
df -h /workspace | tail -1
$PY scripts/train_seg.py "${ARGS[@]}" output_dir="$LED/_preflight" --cfg job --resolve \
  > "$LED"/preflight_config.yaml 2>&1
echo "@@STAGE_0_RC@@ $?"
grep -n 'force_horizon\|force_hist_len\|force_channels\|raw_shape\|max_steps:\|save_every' \
  "$LED"/preflight_config.yaml | head -12

echo "@@STAGE_0b@@ the real dataset instance must hand force windows to the trainer"
CUDA_VISIBLE_DEVICES="" $PY -u scripts/verify_dataset_force.py "${ARGS[@]}" \
  output_dir="$LED/_dsforce" +report="$LED/dataset_force_verify.json" \
  > "$LED"/dataset_force.log 2>&1
echo "@@STAGE_0b_RC@@ $?"
grep -c '"ok": true' "$LED"/dataset_force_verify.json 2>/dev/null

echo "@@STAGE_1@@ train 0 -> 6 (base weight load, save at 6)"
export FW_LEDGER_STAGE=seg1
FW_STOP_STEP=6 FW_WINDOW_LOG=$LED/windows_seg1.log \
$PY -u scripts/train_seg.py "${ARGS[@]}" save_every=6 output_dir="$RUN" resume="$BASE" \
  > "$LED"/s1.log 2>&1
echo "@@STAGE_1_RC@@ $?"
cp "$RUN/config.yaml" "$LED"/config_s1.yaml 2>/dev/null

echo "@@STAGE_2@@ resume 6 -> 12 (light state + FW_BASE_CKPT, save at 12)"
export FW_LEDGER_STAGE=seg2
FW_STOP_STEP=12 FW_WINDOW_LOG=$LED/windows_seg2.log \
$PY -u scripts/train_seg.py "${ARGS[@]}" save_every=12 output_dir="$RUN" \
  resume="$RUN/checkpoints/light/step_000006" > "$LED"/s2.log 2>&1
echo "@@STAGE_2_RC@@ $?"
cp "$RUN/config.yaml" "$LED"/config_s2.yaml 2>/dev/null

echo "@@STAGE_3@@ measure both artifact classes"
find "$RUN/checkpoints/light" -type f -printf '%s\t%p\n' > "$LED"/sizes.txt 2>/dev/null
cat "$LED"/sizes.txt

echo "@@STAGE_4@@ publish a merged checkpoint from the step-12 light state"
export FW_LEDGER_STAGE=export
$PY -u scripts/export_merged_v3.py "${ARGS[@]}" output_dir="$LED" \
  +base_ckpt="$BASE" +light_dir="$RUN/checkpoints/light/step_000012" \
  +output="$MERGED" +record="$PUBREC" > "$LED"/publish.log 2>&1
echo "@@STAGE_4_RC@@ $?"
stat -c '%s\t%n' "$MERGED" >> "$LED"/sizes.txt 2>/dev/null

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

df -h /workspace | tail -1
echo "@@DONE@@"
