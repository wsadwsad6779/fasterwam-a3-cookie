grep -n -E "light_state|save_light_state|load_light_state|adapter" /workspace/FasterWAM/src/fasterwam/r4_patch.py | head -30
echo ===
sed -n "75,100p" /workspace/FasterWAM/src/fasterwam/trainer.py
echo ===
sed -n "585,610p" /workspace/FasterWAM/src/fasterwam/trainer.py