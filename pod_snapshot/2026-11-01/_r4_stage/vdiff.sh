#!/bin/bash
diff /workspace/_r4_stage/r4_eval.sh /workspace/_r4_stage/r4_video.sh > /workspace/_r4_stage/vdiff.txt 2>&1
echo rc=$? >> /workspace/_r4_stage/vdiff.txt
sha256sum /workspace/fasterwam_a3_bridge/bridge/_next/run_matrix.py >> /workspace/_r4_stage/vdiff.txt 2>&1
md5sum /workspace/_r4_stage/r4_video.sh >> /workspace/_r4_stage/vdiff.txt
echo __DONE__ >> /workspace/_r4_stage/vdiff.txt