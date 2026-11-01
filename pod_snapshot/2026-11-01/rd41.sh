cd /workspace/fasterwam_a3_bridge/bridge/_next 2>/dev/null || cd /workspace
find /workspace -name same_column_batch_expert.py 2>/dev/null | head -3
F=$(find /workspace -name same_column_batch_expert.py | head -1)
echo FILE $F
grep -n "def " "$F" | head -60
wc -l "$F"
ls -la /workspace/a3_eval/replay_bundle_dist/
ls /workspace/a3_eval | grep demos
