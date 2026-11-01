cd /workspace/FasterWAM
wc -l scripts/train_light.py scripts/train_light_r5.py src/fasterwam/datasets/lerobot/a3_video_dataset.py src/fasterwam/trainer.py src/fasterwam/r4_patch.py src/fasterwam/r5_patch.py
echo ===
grep -rn --include=*.py -i force src scripts | head -40