ls /workspace/FasterWAM/configs/task
echo ===
grep -rn -E "proprio|state" /workspace/FasterWAM/configs/task/a3_fasterwam.yaml 2>/dev/null | head -10