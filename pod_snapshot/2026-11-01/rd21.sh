cd /workspace
find /workspace/FasterWAM/src/fasterwam -name 'trainer.py' | head
grep -nE 'proprio_encoder|warmup_steps' $(find /workspace/FasterWAM/src/fasterwam -name trainer.py | head -1) | head -20
echo '=== dataset root ==='
sed -n 1,40p /workspace/FasterWAM/configs/data/a3_2cam.yaml
echo '=== lerobot meta ==='
ls /workspace/a3_data/ | head
find /workspace/a3_data -maxdepth 3 -name 'info.json' -o -maxdepth 3 -name 'tasks.parquet' -o -maxdepth 3 -name 'stats.json' | head -10
