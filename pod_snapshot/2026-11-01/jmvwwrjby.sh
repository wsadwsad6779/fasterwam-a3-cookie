cd /workspace/fasterwam_a3_bridge/bridge/_next
wc -l run_matrix.py
grep -n proprio -i run_matrix.py | head -30
echo ===
grep -n force -i run_matrix.py | head -30
echo ===
grep -n "sample[" run_matrix.py | head -20