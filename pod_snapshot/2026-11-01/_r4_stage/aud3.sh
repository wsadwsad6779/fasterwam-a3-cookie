#!/bin/bash
echo "== find r4_patch.py =="
ls -la /workspace/fasterwam_a3_bridge/bridge/_next/r4_patch.py 2>&1
sha256sum /workspace/fasterwam_a3_bridge/bridge/_next/r4_patch.py 2>&1
echo "== grep FW_ env usage =="
grep -rn "FW_SAMPLE_OFFSET|FW_SKIP_STATE|FW_LORA_INIT_SEED|light-state-v1" /workspace/fasterwam_a3_bridge --include=*.py 2>/dev/null | head -20
echo "== trainer resume code =="
grep -rn "resume" /workspace/fasterwam_a3_bridge/FasterWAM/src/fasterwam/trainer.py 2>/dev/null | head -30
echo "== trainer light state load =="
grep -rn "def .*light|optimizer.pt|scheduler.pt|rng.pt|trainer_state" /workspace/fasterwam_a3_bridge/FasterWAM/src/fasterwam/*.py 2>/dev/null | head -20
echo "== DONE =="