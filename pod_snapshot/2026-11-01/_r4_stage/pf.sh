#!/bin/bash
set +e
O=/workspace/_r4_stage/pf.txt
echo === disk === > $O
df -h /workspace /tmp >> $O
echo === frozen dir === >> $O
ls -la /workspace/a3_eval/r4_c0_frozen/ >> $O
du -sh /workspace/a3_eval/r4_c0_frozen/* >> $O
echo === npz keys === >> $O
/opt/venv/bin/python3 /workspace/_r4_stage/keys.py /workspace/a3_eval/r4_c0_frozen/lora_C0_22.npz >> $O 2>&1
echo === video deps === >> $O
/opt/venv/bin/python3 -m pip list 2>/dev/null | grep -Ei imageio|opencv|av|moviepy|Pillow|ffmpeg >> $O
command -v ffmpeg >> $O || echo NO_FFMPEG_BIN >> $O
echo === sim pkg === >> $O
ls /workspace/a3_dual_arm_sim/src/a3_dual_arm_sim/ >> $O 2>&1
echo === r4_eval.sh === >> $O
cat /workspace/_r4_stage/r4_eval.sh >> $O
echo __PFDONE__ >> $O