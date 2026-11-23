#!/bin/bash
LOG=/workspace/fwv1/logs/s1_driver2.log
mkdir -p /workspace/fwv1/logs
echo "[driver2] start $(date -u +%Y-%m-%dT%H:%M:%SZ) (download finished; old pid 148521 is a zombie, wait removed)" | tee -a "$LOG"
unset CL_MAX_STEPS
export HF_ENDPOINT=https://hf-mirror.com
cd /workspace
/opt/venv/bin/python /workspace/_t7_wan_fastdl.py 2>&1 | tee -a "$LOG"
echo "[driver2] fastdl rc=${PIPESTATUS[0]}" | tee -a "$LOG"
/opt/venv/bin/python /workspace/_t7_p4_smoke.py 2>&1 | tee -a "$LOG"
echo "[driver2] smoke rc=${PIPESTATUS[0]}" | tee -a "$LOG"
/opt/venv/bin/python /workspace/_t7_p7_verify.py 2>&1 | tee -a "$LOG"
echo "[driver2] p7 rc=${PIPESTATUS[0]}" | tee -a "$LOG"
echo "[driver2] ALL DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG"
exec bash