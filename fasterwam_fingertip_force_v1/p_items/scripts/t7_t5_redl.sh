#!/bin/bash
OUT=/workspace/fwv1/wan_base/models_t5_umt5-xxl-enc-bf16.pth
LOG=/workspace/fwv1/logs/t5_redownload.log
URL='https://modelscope.cn/api/v1/models/Wan-AI/Wan2.2-TI2V-5B/repo?Revision=master&FilePath=models_t5_umt5-xxl-enc-bf16.pth'
curl -sL --retry 8 --retry-all-errors -C - -o "$OUT" "$URL" > "$LOG" 2>&1
echo "DONE_T5_$?" >> "$LOG"
ls -la "$OUT" >> "$LOG"