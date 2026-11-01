cd /workspace/FasterWAM/src/fasterwam
sed -n "195,240p" datasets/lerobot/robot_video_dataset.py
echo ===TRAINER 300-360,
sed -n "300,318p" trainer.py
sed -n "344,372p" trainer.py