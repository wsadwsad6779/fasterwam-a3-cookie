cd /workspace/_r4_stage
ls -1t | head -25
echo GREP
grep -rln run_matrix /workspace/_r4_stage /workspace/_r5_stage 2>/dev/null | head -12
echo HEADS
grep -rnE 'run_matrix|PYTHONPATH|cd /workspace' /workspace/_r5_stage/*.sh 2>/dev/null | head -20
echo MD5PAIR
md5sum /workspace/fasterwam_a3_bridge/bridge/_next/run_matrix.py /workspace/fasterwam-a3-cookie/bridge/_next/run_matrix.py
echo DONE