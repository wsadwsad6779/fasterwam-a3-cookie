#!/bin/bash
# Single-shot guarded launcher for drive_diverse_20261119.sh (v3 training).
# Guards: no lock file, no running training/eval process, no prior training evidence.
# This wrapper exists because the drive script itself has no lock/resume
# (audit D3 in V3_PIPELINE_COMPLIANCE_AUDIT_20261119.md). Launch exactly once.
set -u
cd /workspace/eval_restore_20261117/reconstructed_normalizer || exit 1
WORK=/root/repair_runs/action_expert_v3
LOCK="$WORK/.launch.lock"
LOG=drive_diverse.log

# 1) prior-launch evidence (any of these means the pipeline already ran or is running)
if [ -e "$LOCK" ]; then echo "ABORT_LOCK_EXISTS $LOCK"; exit 2; fi
if [ -f "$LOG" ] && grep -q "DRIVE_START" "$LOG"; then echo "ABORT_DRIVE_LOG_EXISTS"; exit 2; fi
if [ -f "$WORK/segments.jsonl" ]; then echo "ABORT_SEGMENTS_EXISTS"; exit 2; fi
if [ -f "$WORK/last.pt" ]; then echo "ABORT_LASTPT_EXISTS"; exit 2; fi

# 2) running-process guard
RUN=$(pgrep -af "train_action_expert_segmented|drive_diverse_20261119|c_class_eval_20261119" || true)
if [ -n "$RUN" ]; then echo "ABORT_PROCESS_RUNNING"; echo "$RUN"; exit 2; fi

# 3) data presence guard
if [ ! -f "teacher_diverse_20261119/manifest.json" ]; then echo "ABORT_NO_MANIFEST"; exit 2; fi
if [ ! -f "teacher_diverse_20261119/acceptance.json" ]; then echo "ABORT_NO_ACCEPTANCE_JSON (run accept_diverse_corpus_20261119.py first)"; exit 2; fi
PASS=$(/opt/venv/bin/python -c "import json;print(bool(json.load(open('teacher_diverse_20261119/acceptance.json'))['pass']))")
if [ "$PASS" != "True" ]; then echo "ABORT_ACCEPTANCE_NOT_PASS"; exit 2; fi

# 4) clean environment (memory: terminal env persists across commands)
unset SEG_STEPS TRAIN_RESUME TRAIN_WORK TRAIN_PUB TRAIN_DATA_DIR TOTAL_TARGET TRAIN_LR \
      C_CKPT C_OUT C_SEEDS C_RESUME C_MAX_STEPS C_WRIST_ABORT_N C_PAD_ABORT_N CL_MAX_STEPS \
      DIV_OUT DIV_STEPS DIV_EP_TIMEOUT DIV_SPEC 2>/dev/null || true

mkdir -p "$WORK"
echo "pid=$$ started=$(date -u +%FT%TZ)" > "$LOCK"
nohup bash drive_diverse_20261119.sh > /dev/null 2>&1 &
DRV=$!
echo "LAUNCHED wrapper_pid=$$ drive_pid=$DRV started=$(date -u +%FT%TZ)"
