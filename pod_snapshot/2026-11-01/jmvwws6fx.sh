cd /workspace/fasterwam_a3_bridge/bridge/_next
grep -n "def |obs|sample" common.py | head -40
echo ===
wc -l common.py
find /workspace/fasterwam_a3_bridge -name a3_wrap.py | head