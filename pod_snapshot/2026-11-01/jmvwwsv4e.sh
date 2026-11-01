find /workspace/fasterwam_a3_bridge /workspace/fasterwam-a3-cookie -name infer_a3_worker.py 2>/dev/null | head
echo ===
grep -n -E "proprio|observation|force" /workspace/fasterwam_a3_bridge/bridge/infer_a3_worker.py 2>/dev/null | head -40