sed -n "55,100p" /workspace/fasterwam_a3_bridge/bridge/_next/common.py
echo ===A
grep -n -E "FORCE_KEY|L_INNER|L_OUTER|WRIST" /workspace/fasterwam_a3_bridge/bridge/_next/safety_wrap.py | head -20