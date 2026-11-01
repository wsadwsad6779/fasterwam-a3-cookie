grep -n -E "FW_[A-Z_]+" /workspace/FasterWAM/src/fasterwam/r4_patch.py /workspace/FasterWAM/src/fasterwam/r5_patch.py /workspace/FasterWAM/src/fasterwam/trainer.py | head -30
echo ===
grep -rn -E "os.environ|getenv" /workspace/FasterWAM/src/fasterwam/datasets /workspace/FasterWAM/src/fasterwam/runtime.py | head -20