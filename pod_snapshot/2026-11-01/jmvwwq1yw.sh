cd /workspace/FasterWAM/src/fasterwam
grep -rn --include=*.py proprio datasets trainer.py r4_patch.py r5_patch.py runtime.py | head -40
echo ===
ls datasets/lerobot/processors