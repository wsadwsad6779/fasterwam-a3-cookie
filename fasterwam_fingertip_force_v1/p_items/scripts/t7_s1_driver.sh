#!/bin/bash
LOG=/workspace/fwv1/logs/s1_driver.log
mkdir -p /workspace/fwv1/logs
echo "[driver] start $(date -u +%Y-%m-%dT%H:%M:%SZ); waiting for pid 148521 (do not restart live download)" | tee -a "$LOG"
unset CL_MAX_STEPS
export HF_ENDPOINT=https://hf-mirror.com
while kill -0 148521 2>/dev/null; do
  echo "[driver] $(date -u +%H:%M:%SZ) download alive; wan_base=$(du -sh /workspace/fwv1/wan_base 2>/dev/null | cut -f1)" >> "$LOG"
  sleep 30
done
echo "[driver] old download exited at $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG"
cd /workspace
/opt/venv/bin/python /workspace/_t7_wan_fastdl.py 2>&1 | tee -a "$LOG"
echo "[driver] fastdl rc=${PIPESTATUS[0]}" | tee -a "$LOG"
/opt/venv/bin/python /workspace/_t7_p4_smoke.py 2>&1 | tee -a "$LOG"
echo "[driver] smoke rc=${PIPESTATUS[0]}" | tee -a "$LOG"
/opt/venv/bin/python /workspace/_t7_p7_verify.py 2>&1 | tee -a "$LOG"
echo "[driver] p7 rc=${PIPESTATUS[0]}" | tee -a "$LOG"
echo "[driver] ALL DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG"
exec bash