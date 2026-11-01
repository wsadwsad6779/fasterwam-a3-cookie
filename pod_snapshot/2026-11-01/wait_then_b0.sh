#!/usr/bin/env bash
source /workspace/activate_fasterwam_a3.sh
B=/workspace/FasterWAM/checkpoints
T5=$B/Wan-AI/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth
L=/workspace/a3_train_runs/a3_lora_r1/step_002000_merged.pt
for i in $(seq 1 240); do
  if grep -q MS_DONE /workspace/dl_ms.log && grep -q LORA_DONE /workspace/dl_lora.log; then echo GATE_DL_DONE; break; fi
  sleep 30
done
sz(){ stat -c %s $1 2>/dev/null; }
echo T5=$(sz $T5)/11361920418 LORA=$(sz $L)/11117977659
for i in $(seq 1 20); do [ "$(sz $T5)" = 11361920418 ] && [ "$(sz $L)" = 11117977659 ] && break; sleep 30; done
echo GATE_SIZES T5=$(sz $T5) LORA=$(sz $L)
bash /workspace/run_b0.sh > /workspace/b0.log 2>&1
echo B0_RC_$? >> /workspace/b0.log
